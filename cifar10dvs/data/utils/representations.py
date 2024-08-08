import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch as th


class RepresentationBase(ABC):
    @abstractmethod
    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        ...

    @abstractmethod
    def get_shape(self) -> Tuple[int, int, int]:
        ...

    @staticmethod
    @abstractmethod
    def get_numpy_dtype() -> np.dtype:
        ...

    @staticmethod
    @abstractmethod
    def get_torch_dtype() -> th.dtype:
        ...

    @property
    def dtype(self) -> th.dtype:
        return self.get_torch_dtype()

    @staticmethod
    def _is_int_tensor(tensor: th.Tensor) -> bool:
        return not th.is_floating_point(tensor) and not th.is_complex(tensor)


class StackedHistogram(RepresentationBase):
    def __init__(self, bins: int, height: int, width: int, count_cutoff: Optional[int] = None, fast_mode: bool = True):
        """
        In case of fast-mode == True: use uint8 to construct the representation, but could lead to overflow.
        In case of fast-mode == False: use int16 to construct the representation, and convert to uint8 after clipping.

        Note: Overflow should not be a big problem because it happens only for hot pixels. In case of overflow,
        the value will just start accumulating from 0 again.
        """
        assert bins >= 1
        self.bins = bins
        assert height >= 1
        self.height = height
        assert width >= 1
        self.width = width
        self.count_cutoff = count_cutoff
        if self.count_cutoff is None:
            self.count_cutoff = 255
        else:
            assert count_cutoff >= 1
            self.count_cutoff = min(count_cutoff, 255)
        self.fast_mode = fast_mode
        self.channels = 2

    @staticmethod
    def get_numpy_dtype() -> np.dtype:
        return np.dtype('uint8')

    @staticmethod
    def get_torch_dtype() -> th.dtype:
        return th.uint8

    def merge_channel_and_bins(self, representation: th.Tensor):
        assert representation.dim() == 4
        return th.reshape(representation, (-1, self.height, self.width))

    def get_shape(self) -> Tuple[int, int, int]:
        return 2 * self.bins, self.height, self.width

    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        device = x.device
        assert y.device == pol.device == time.device == device
        assert self._is_int_tensor(x)
        assert self._is_int_tensor(y)
        assert self._is_int_tensor(pol)
        assert self._is_int_tensor(time)

        dtype = th.uint8 if self.fast_mode else th.int16

        representation = th.zeros((self.channels, self.bins, self.height, self.width),
                                  dtype=dtype, device=device, requires_grad=False)

        if x.numel() == 0:
            assert y.numel() == 0
            assert pol.numel() == 0
            assert time.numel() == 0
            return self.merge_channel_and_bins(representation.to(th.uint8))
        assert x.numel() == y.numel() == pol.numel() == time.numel()

        assert pol.min() >= 0
        assert pol.max() <= 1

        bn, ch, ht, wd = self.bins, self.channels, self.height, self.width

        # NOTE: assume sorted time
        t0_int = time[0]
        t1_int = time[-1]
        assert t1_int >= t0_int
        t_norm = time - t0_int
        t_norm = t_norm / max((t1_int - t0_int), 1)
        t_norm = t_norm * bn
        t_idx = t_norm.floor()
        t_idx = th.clamp(t_idx, max=bn - 1)

        indices = (x.long() +
                   wd * y.long() +
                   ht * wd * t_idx.long() +
                   bn * ht * wd * pol.long())
        indices = torch.clamp(indices, max=bn * ht * wd * ch - 1)
        values = th.ones_like(indices, dtype=dtype, device=device)
        representation.put_(indices, values, accumulate=True)
        representation = th.clamp(representation, min=0, max=self.count_cutoff)
        if not self.fast_mode:
            representation = representation.to(th.uint8)

        return representation
        # return self.merge_channel_and_bins(representation)

class IntegratedFixedFrameNumber:
    def __init__(self, H: int, W: int, frames_num: int, split_by: str = "number"):
        self.frames_num = frames_num
        self.split_by = split_by
        self.H = H
        self.W = W
        self.device = None

    def integrate_events_segment_to_frame(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, j_l: int = 0, j_r: int = -1) -> th.Tensor:
        """
        :param pol:
        :param y:
        :param x:
        :param events: a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``
        :type events: Dict
        :param H: height of the frame
        :type H: int
        :param W: weight of the frame
        :type W: int
        :param j_l: the start index of the integral interval, which is included
        :type j_l: int
        :param j_r: the right index of the integral interval, which is not included
        :type j_r:
        :return: frames
        :rtype: np.ndarray

        Denote a two channels frame as :math:`F` and a pixel at :math:`(p, x, y)` as :math:`F(p, x, y)`, the pixel value is integrated from the events data whose indices are in :math:`[j_{l}, j_{r})`:

    .. math::

        F(p, x, y) &= \sum_{i = j_{l}}^{j_{r} - 1} \mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})

    where :math:`\lfloor \cdot \rfloor` is the floor operation, :math:`\mathcal{I}_{p, x, y}(p_{i}, x_{i}, y_{i})` is an indicator function and it equals 1 only when :math:`(p, x, y) = (p_{i}, x_{i}, y_{i})`.
        """
        # 累计脉冲需要用bitcount而不能直接相加，原因可参考下面的示例代码，以及
        # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
        # We must use ``bincount`` rather than simply ``+``. See the following reference:
        # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments

        # Here is an example:

        # height = 3
        # width = 3
        # frames = np.zeros(shape=[2, height, width])
        # events = {
        #     'x': np.asarray([1, 2, 1, 1]),
        #     'y': np.asarray([1, 1, 1, 2]),
        #     'p': np.asarray([0, 1, 0, 1])
        # }
        #
        # frames[0, events['y'], events['x']] += (1 - events['p'])
        # frames[1, events['y'], events['x']] += events['p']
        # print('wrong accumulation\n', frames)
        #
        # frames = np.zeros(shape=[2, height, width])
        # for i in range(events['p'].__len__()):
        #     frames[events['p'][i], events['y'][i], events['x'][i]] += 1
        # print('correct accumulation\n', frames)
        #
        # frames = np.zeros(shape=[2, height, width])
        # frames = frames.reshape(2, -1)
        #
        # mask = [events['p'] == 0]
        # mask.append(np.logical_not(mask[0]))
        # for i in range(2):
        #     position = events['y'][mask[i]] * width + events['x'][mask[i]]
        #     events_number_per_pos = np.bincount(position)
        #     idx = np.arange(events_number_per_pos.size)
        #     frames[i][idx] += events_number_per_pos
        # frames = frames.reshape(2, height, width)
        # print('correct accumulation by bincount\n', frames)

        frame = th.zeros((2, self.H * self.W), device=self.device)
        x = x
        y = y
        p = pol[j_l: j_r]
        mask = []
        mask.append(p == 0)
        mask.append(th.logical_not(mask[0]))
        for c in range(2):
            position = y[j_l: j_r][mask[c]] * self.W + x[j_l: j_r][mask[c]]
            events_number_per_pos = th.bincount(position)
            frame[c][th.arange(len(events_number_per_pos))] += events_number_per_pos
        return frame.reshape((2, self.H, self.W))

    def cal_fixed_frames_number_segment_index(self, t) -> tuple:
        """
        :param t:
        :param events_t: events' t
        :type events_t: numpy.ndarray
        :param split_by: 'time' or 'number'
        :type split_by: str
        :param frames_num: the number of frames
        :type frames_num: int
        :return: a tuple ``(j_l, j_r)``
        :rtype: tuple

        Denote ``frames_num`` as :math:`M`, if ``split_by`` is ``'time'``, then

        .. math::

            \\Delta T & = [\\frac{t_{N-1} - t_{0}}{M}] \\\\
            j_{l} & = \\mathop{\\arg\\min}\\limits_{k} \\{t_{k} | t_{k} \\geq t_{0} + \\Delta T \\cdot j\\} \\\\
            j_{r} & = \\begin{cases} \\mathop{\\arg\\max}\\limits_{k} \\{t_{k} | t_{k} < t_{0} + \\Delta T \\cdot (j + 1)\\} + 1, & j <  M - 1 \\cr N, & j = M - 1 \\end{cases}

        If ``split_by`` is ``'number'``, then

        .. math::
            j_{l} & = [\\frac{N}{M}] \\cdot j \\\\
            j_{r} & = \\begin{cases} [\\frac{N}{M}] \\cdot (j + 1), & j <  M - 1 \\cr N, & j = M - 1 \\end{cases}
        """
        j_l = np.zeros(shape=[self.frames_num], dtype=int)
        j_r = np.zeros(shape=[self.frames_num], dtype=int)
        N = len(t)

        if self.split_by == 'number':
            di = N // self.frames_num
            for i in range(self.frames_num):
                j_l[i] = i * di
                j_r[i] = j_l[i] + di
            j_r[-1] = N

        elif self.split_by == 'time':
            dt = (t[-1] - t[0]) // self.frames_num
            idx = np.arange(N)
            for i in range(self.frames_num):
                t_l = dt * i + t[0]
                t_r = t_l + dt
                mask = np.logical_and(t >= t_l, t < t_r)
                idx_masked = idx[mask]
                j_l[i] = idx_masked[0]
                j_r[i] = idx_masked[-1] + 1

            j_r[-1] = N
        else:
            raise NotImplementedError

        return j_l, j_r

    def integrate_events_by_fixed_frames_number(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        """
        :param time:
        :param pol:
        :param y:
        :param x:
        :param events: a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``; np.load(xxx.npz)
        :type events: Dict
        :param split_by: 'time' or 'number'
        :type split_by: str
        :param frames_num: the number of frames
        :type frames_num: int
        :param H: the height of frame
        :type H: int
        :param W: the weight of frame
        :type W: int
        :return: frames
        :rtype: np.ndarray

        Integrate events to frames by fixed frames number. See ``cal_fixed_frames_number_segment_index`` and ``integrate_events_segment_to_frame`` for more details.
        """
        j_l, j_r = self.cal_fixed_frames_number_segment_index(time)
        frames = th.zeros((self.frames_num, 2, self.H, self.W), device=self.device)
        for i in range(self.frames_num):
            frames[i] = self.integrate_events_segment_to_frame(x, y, pol, j_l[i], j_r[i])
        return frames  # [self.frames_num, 2, H, W]

    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        self.device = x.device
        return self.integrate_events_by_fixed_frames_number(x, y, pol, time)


def get_file_list(file_path: str) -> list:
    return os.listdir(file_path)


def merge_channel_and_bins(representation: th.Tensor):
    assert representation.dim() == 4
    return th.reshape(representation, (-1, 200, 300))
