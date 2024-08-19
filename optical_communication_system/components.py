from collections.abc import Callable, Sequence
import math
from typing import Any, Optional
import typing

import tensorflow as tf

__all__ = [
    "Timing",
    "ElectricalSignal",
    "OpticalSignal",

    "PAMEncoder",
    "Transmitter",
    "IntensityModulator",
    "ElectricalAWGN",
    "OpticalAWGN",
    "OpticalLoss",
    "ChromaticDispersion",
    "OpticalFiber",
    "ElectricalLowPassFilter",
    "Photodiode",
    "PAMDecoder",
    "BERAnalyzer",
    "BERTester",
    "System"
]


class Constants:
    #: Speed of light in vacuum in m/s
    c = tf.Variable(tf.constant(2.99792458E8))

    pi = tf.Variable(tf.constant(math.pi))
    e = tf.Variable(tf.constant(math.e))


class Timing(tf.experimental.ExtensionType):
    #: Time axis of the signal (in seconds)
    time: tf.Tensor  # shape=[None]; dtype=tf.float32

    #: Sampling rate in Hz
    sampling_rate: tf.Tensor  # shape=[]; dtype=tf.float32

    #: Data rate in symbols/second
    symbol_rate: tf.Tensor  # shape=[]; dtype=tf.float32


timing_spec = Timing.Spec(
    time=tf.TensorSpec(shape=[None], dtype=tf.float32),
    sampling_rate=tf.TensorSpec(shape=[], dtype=tf.float32),
    symbol_rate=tf.TensorSpec(shape=[], dtype=tf.float32)
)


class ElectricalSignal(tf.experimental.ExtensionType):
    #: Signal's timing information
    timing: Timing

    #: Signal's samples
    samples: tf.Tensor  # shape=[None]; dtype=tf.complex64


electrical_signal_spec = ElectricalSignal.Spec(
    timing=timing_spec,
    samples=tf.TensorSpec(shape=[None], dtype=tf.complex64)
)


class IQSignal(tf.experimental.ExtensionType):
    i: ElectricalSignal
    q: ElectricalSignal


iq_signal_spec = IQSignal.Spec(
    i=electrical_signal_spec,
    q=electrical_signal_spec
)


class OpticalSignal(tf.experimental.ExtensionType):
    #: Signal's timing information
    timing: Timing

    #: Signal's frequencies (in Hz), for every frequency there's an amplitude
    frequencies: tf.Tensor  # shape=[None]; dtype=tf.float32

    #: Signal's samples
    samples: tf.Tensor  # shape=[None]; dtype=tf.complex64

    #: Power (maximum) of the optical signal in watts
    power: tf.Tensor  # shape=[]; dtype=tf.float32

    #: Central frequency of the of the emitted light in Hz
    central_frequency: tf.Tensor  # shape=[]; dtype=tf.float32


optical_signal_spec = OpticalSignal.Spec(
    timing=timing_spec,
    frequencies=tf.TensorSpec(shape=[None], dtype=tf.float32),
    samples=tf.TensorSpec(shape=[None, None], dtype=tf.complex64),
    power=tf.TensorSpec(shape=[], dtype=tf.float32),
    central_frequency=tf.TensorSpec(shape=[], dtype=tf.float32)
)

bit_dtype = tf.int32
bit_sequence_spec = tf.TensorSpec(shape=[None], dtype=bit_dtype)


class PAMConstellation(tf.Module):
    order: tf.Variable  # shape=(1,); dtype=int32
    mapping: tf.Variable  # shape=(2 ** order,); dtype=complex64

    def __init__(
        self,
        *,
        order: int,
        mapping: Optional[Sequence[float]] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.order = tf.Variable(tf.constant(order))

        if mapping is None:
            self.mapping = tf.Variable(
                typing.cast(tf.Tensor, self._default_pam_constellation())
            )
        else:
            self.mapping = tf.Variable(tf.constant(mapping))

    @tf.function
    def _default_pam_constellation(self) -> tf.Tensor:
        return tf.vectorized_map(
            fn=lambda i: tf.cast(i / (tf.math.pow(2, self.order) - 1), tf.complex64),
            elems=tf.range(tf.math.pow(2, self.order))
        )

    @tf.function(input_signature=(bit_sequence_spec,))
    def encode_symbol(self, symbol: tf.Tensor) -> tf.Tensor:
        symbol_index: Any = tf.reduce_sum(
            symbol * 2 ** (self.order - tf.range(self.order) - 1)
        )

        return self.mapping[symbol_index]

    @tf.function(input_signature=(tf.TensorSpec(shape=[], dtype=tf.complex64),))
    def decode_symbol(self, symbol: tf.Tensor) -> tf.Tensor:
        best_index = tf.cast(tf.argmin(tf.abs(self.mapping - symbol)), tf.int32)

        return tf.vectorized_map(
            fn=lambda i: tf.cast(
                tf.math.floormod(best_index / 2 ** (self.order - i - 1), 2),
                bit_dtype
            ),
            elems=tf.range(self.order)
        )


class PAMEncoder(tf.Module):
    #: Sampling rate in Hz
    sampling_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Constellation used for encoding
    constellation: PAMConstellation

    #: Modulation order (number of bits per symbol)
    order: tf.Variable  # shape=(1,); dtype=int32

    def __init__(
        self,
        sampling_rate: float,
        data_rate: float,
        order: int = 1,
        constellation: Optional[PAMConstellation] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        if constellation is None:
            constellation = PAMConstellation(order=order)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.order = tf.Variable(tf.constant(order))
        self.constellation = constellation

    @tf.function(input_signature=(bit_sequence_spec,))
    def __call__(self, bitseq: tf.Tensor) -> ElectricalSignal:
        tf.debugging.assert_equal(tf.math.floormod(tf.size(bitseq), self.order), 0)

        signal_duration = tf.divide(
            tf.cast(tf.size(bitseq), tf.float32),
            self.data_rate
        )

        time = tf.linspace(
            0.0,
            signal_duration,
            tf.cast(tf.round(self.sampling_rate * signal_duration), tf.int32)
        )

        pam_sequence = tf.vectorized_map(
            fn=lambda i: self.constellation.encode_symbol(bitseq[i * self.order:(i + 1) * self.order]),
            elems=tf.range(tf.math.floordiv(tf.size(bitseq), self.order))
        )

        return ElectricalSignal(
            timing=Timing(
                time=time,
                sampling_rate=self.sampling_rate,
                symbol_rate=tf.divide(self.data_rate, tf.cast(self.order, tf.float32))
            ),
            samples=tf.repeat(
                pam_sequence,
                tf.cast(
                    tf.math.ceil(tf.divide(tf.size(time), tf.size(pam_sequence))),
                    tf.int32
                )
            )[:tf.size(time)]
        )


class IQConstellation(tf.Module):
    order: tf.Variable  # shape=(1,); dtype=int32
    mapping: tf.Variable  # shape=(2 ** order,2); dtype=complex64

    def __init__(
        self,
        *,
        order: int,
        mapping: Sequence[Sequence[float]],
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.order = tf.Variable(tf.constant(order))
        self.mapping = tf.Variable(tf.constant(mapping))

    @tf.function(input_signature=(bit_sequence_spec,))
    def encode_symbol(self, symbol: tf.Tensor) -> tf.Tensor:
        symbol_index: Any = tf.reduce_sum(
            symbol * 2 ** (self.order - tf.range(self.order) - 1)
        )

        return self.mapping[symbol_index]

    @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
    def decode_symbol(self, symbol: tf.Tensor) -> tf.Tensor:
        best_index = tf.cast(
            tf.argmin(
                tf.reduce_sum(
                    tf.abs(self.mapping - symbol),
                    axis=1
                )
            ),
            tf.int32
        )

        return tf.vectorized_map(
            fn=lambda i: tf.cast(
                tf.math.floormod(best_index / 2 ** (self.order - i - 1), 2),
                bit_dtype
            ),
            elems=tf.range(self.order)
        )


class IQEncoder(tf.Module):
    #: Sampling rate in Hz
    sampling_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Constellation used for encoding
    constellation: IQConstellation

    #: Modulation order (number of bits per symbol)
    order: tf.Variable  # shape=(1,); dtype=int32

    def __init__(
        self,
        *,
        sampling_rate: float,
        data_rate: float,
        constellation: IQConstellation,
        order: int = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.order = tf.Variable(tf.constant(order))
        self.constellation = constellation

    @tf.function(input_signature=(bit_sequence_spec,))
    def __call__(self, bitseq: tf.Tensor) -> IQSignal:
        tf.debugging.assert_equal(tf.math.floormod(tf.size(bitseq), self.order), 0)

        signal_duration = tf.divide(
            tf.cast(tf.size(bitseq), tf.float32),
            self.data_rate
        )

        time = tf.linspace(
            0.0,
            signal_duration,
            tf.cast(tf.round(self.sampling_rate * signal_duration), tf.int32)
        )

        iq_sequence = tf.vectorized_map(
            fn=lambda i: self.constellation.encode_symbol(bitseq[i * self.order:(i + 1) * self.order]),
            elems=tf.range(tf.math.floordiv(tf.size(bitseq), self.order))
        )

        timing = Timing(
            time=time,
            sampling_rate=self.sampling_rate,
            symbol_rate=tf.divide(self.data_rate, tf.cast(self.order, tf.float32))
        )

        repeat_count = tf.cast(
            tf.math.ceil(tf.divide(tf.size(time), tf.shape(iq_sequence)[0])),
            tf.int32
        )

        i_samples = tf.repeat(iq_sequence[:,0], repeat_count)[:tf.size(time)]
        q_samples = tf.repeat(iq_sequence[:,1], repeat_count)[:tf.size(time)]

        return IQSignal(
            i=ElectricalSignal(
                timing=timing,
                samples=i_samples
            ),
            q=ElectricalSignal(
                timing=timing,
                samples=q_samples
            ),
        )


@tf.function
def gaussian_distribution(x: tf.Tensor, mean: tf.Tensor, variance: tf.Tensor) -> tf.Tensor:
    return 1 / (
        variance * tf.sqrt(tf.multiply(2.0, Constants.pi))
    ) * Constants.e ** (-0.5 * tf.subtract(x, mean) ** 2 / variance ** 2)


class Transmitter(tf.Module):
    #: Output power in watts
    power: tf.Variable  # shape=(1,); dtype=float32

    #: Central frequency of the of the emitted light in Hz
    central_frequency: tf.Variable  # shape=(1,); dtype=float32

    #: Line width of the signal, full width at half-maximum (FWHM), Hz
    linewidth: tf.Variable  # shape=(1,); dtype=float32

    #: Number of frequency components in the signal
    frequency_count: tf.Variable  # shape=(1,); dtype=int32

    def __init__(
        self,
        *,
        power: float = 2e-3,
        central_frequency: float = 193.414e12,
        linewidth: float = 1e6,
        frequency_count: int = 128,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.power = tf.Variable(tf.constant(power))
        self.central_frequency = tf.Variable(tf.constant(central_frequency))
        self.linewidth = tf.Variable(tf.constant(linewidth))
        self.frequency_count = tf.Variable(tf.constant(frequency_count))

    @tf.function(input_signature=(timing_spec,))
    def __call__(self, timing: Timing) -> OpticalSignal:
        gauss_variance = 1.0 / (2.0 * tf.math.log(2.0)) ** 0.5 * self.linewidth

        frequencies = tf.linspace(
            self.central_frequency - tf.multiply(self.linewidth, 10.0),
            self.central_frequency + tf.multiply(self.linewidth, 10.0),
            self.frequency_count
        )

        distribution = gaussian_distribution(
            frequencies,
            self.central_frequency,
            gauss_variance
        )
        distribution /= tf.reduce_sum(distribution)

        samples = tf.reshape(
            tf.tile(
                tf.cast(tf.math.sqrt(self.power * distribution), tf.complex64),
                [tf.size(timing.time)],
            ),
            [tf.size(timing.time), self.frequency_count]
        )

        return OpticalSignal(
            timing=timing,
            frequencies=frequencies,
            samples=samples,
            central_frequency=self.central_frequency,
            power=self.power
        )


class IntensityModulator(tf.Module):
    #: Carrier signal source
    carrier_source: Callable[[Timing], OpticalSignal]

    def __init__(self, carrier_source: Callable, name=None) -> None:
        super().__init__(name=name)

        self.carrier_source = carrier_source

    @tf.function(input_signature=(electrical_signal_spec,))
    def __call__(self, modulation_signal: ElectricalSignal) -> OpticalSignal:
        carrier_signal = self.carrier_source(modulation_signal.timing)

        modulated_signal = tf.multiply(
            carrier_signal.samples,
            tf.cast(
                tf.abs(
                    tf.reshape(
                        tf.repeat(
                            modulation_signal.samples,
                            tf.size(carrier_signal.frequencies)
                        ),
                        tf.shape(carrier_signal.samples)
                    )
                ) ** 0.5,
                tf.complex64
            )
        )

        return OpticalSignal(
            timing=carrier_signal.timing,
            frequencies=carrier_signal.frequencies,
            samples=modulated_signal,
            central_frequency=carrier_signal.central_frequency,
            power=carrier_signal.power
        )


class OpticalPhaseShifter(tf.Module):
    phase_shift: tf.Variable  # shape=(1,); dtype=float32

    def __init__(self, *, phase_shift: float, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.phase_shift = tf.Variable(tf.constant(phase_shift))

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, input_signal: OpticalSignal) -> OpticalSignal:
        shifted_samples = input_signal.samples * tf.math.exp(tf.complex(0.0, self.phase_shift))

        return OpticalSignal(
            timing=input_signal.timing,
            frequencies=input_signal.frequencies,
            samples=shifted_samples,
            central_frequency=input_signal.central_frequency,
            power=input_signal.power
        )


class IQModulator(tf.Module):
    def __init__(self, carrier_source: Callable, name=None) -> None:
        super().__init__(name=name)

        self.i_modulator = IntensityModulator(carrier_source)
        self.q_modulator = IntensityModulator(
            carrier_source=System(components=[
                carrier_source,
                OpticalPhaseShifter(phase_shift=0.5 * math.pi)
            ])
        )

    @tf.function(input_signature=(iq_signal_spec,))
    def __call__(self, iq_signal: IQSignal) -> OpticalSignal:
        i_signal = self.i_modulator(iq_signal.i)
        q_signal = self.q_modulator(iq_signal.q)

        combined_samples = (i_signal.samples + q_signal.samples) * (0.5 ** 0.5)

        return OpticalSignal(
            timing=iq_signal.timing,
            frequencies=iq_signal.frequencies,
            samples=combined_samples,
            central_frequency=iq_signal.central_frequency,
            power=iq_signal.power
        )


class ElectricalAWGN(tf.Module):
    #: Noise level in dB relative to the input signal
    level: tf.Variable  # shape=(1,); dtype=float32

    #: Fixed amount of noise in watts that is independent of input signal
    fixed_power: tf.Variable  # shape=(1,); dtype=float32

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        *,
        level: float,
        fixed_power: float = 0.0,
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.level = tf.Variable(tf.constant(level))
        self.fixed_power = tf.Variable(tf.constant(fixed_power))

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

    @tf.function(input_signature=(electrical_signal_spec,))
    def __call__(self, input_signal: ElectricalSignal) -> ElectricalSignal:
        noise_amplitude = (tf.add(
            self.fixed_power,
            input_signal.power * tf.math.pow(10.0, self.level * 0.1)
        ) * 0.5) ** 0.5  # * 0.5 to account for two noise components

        noise_real = self.rng.normal(
            shape=tf.shape(input_signal.samples),
            mean=0.0,
            stddev=noise_amplitude,
            dtype=tf.float32
        )

        noise_imag = self.rng.normal(
            shape=tf.shape(input_signal.samples),
            mean=0.0,
            stddev=noise_amplitude,
            dtype=tf.float32
        )

        noise = tf.dtypes.complex(noise_real, noise_imag)

        return ElectricalSignal(
            timing=input_signal.timing,
            samples=tf.add(noise, input_signal.samples)
        )


class OpticalAWGN(tf.Module):
    #: Noise level in dB relative to the input signal
    level: tf.Variable  # shape=(1,); dtype=float32

    #: Fixed amount of noise in watts that is independent of input signal
    fixed_power: tf.Variable  # shape=(1,); dtype=float32

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        *,
        level: float,
        fixed_power: float = 0.0,
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.level = tf.Variable(tf.constant(level))
        self.fixed_power = tf.Variable(tf.constant(fixed_power))

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, input_signal: OpticalSignal) -> OpticalSignal:
        noise_amplitude = tf.divide(
            tf.add(
                self.fixed_power,
                input_signal.power * tf.math.pow(10.0, self.level * 0.1)
            ) * 0.5,  # * 0.5 to account for two noise components
            tf.cast(tf.size(input_signal.frequencies), tf.float32)
        ) ** 0.5

        noise_real = self.rng.normal(
            shape=tf.shape(input_signal.samples),
            mean=0.0,
            stddev=noise_amplitude,
            dtype=tf.float32
        )

        noise_imag = self.rng.normal(
            shape=tf.shape(input_signal.samples),
            mean=0.0,
            stddev=noise_amplitude,
            dtype=tf.float32
        )

        noise = tf.reshape(
            tf.dtypes.complex(noise_real, noise_imag),
            tf.shape(input_signal.samples)
        )

        return OpticalSignal(
            timing=input_signal.timing,
            samples=tf.add(noise, input_signal.samples),
            frequencies=input_signal.frequencies,
            central_frequency=input_signal.central_frequency,
            power=input_signal.power
        )


class OpticalLoss(tf.Module):
    #: Optical signal power loss in dB
    loss: tf.Variable  # shape=(1,); dtype=float32

    def __init__(self, *, loss: float, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.loss = tf.Variable(tf.constant(loss))

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
        power_loss_coeff = tf.math.pow(10.0, self.loss * 0.1)
        samples = tf.divide(
            signal.samples,
            tf.cast(tf.sqrt(power_loss_coeff), tf.complex64)
        )

        return OpticalSignal(
            timing=signal.timing,
            frequencies=signal.frequencies,
            samples=samples,
            central_frequency=signal.central_frequency,
            power=signal.power / power_loss_coeff
        )


class ChromaticDispersion(tf.Module):
    #: Chromatic dispersion coefficient (ps / (nm * km))
    coefficient: tf.Variable  # shape=(1,); dtype=float32
    distance: tf.Variable  # shape=(1,); dtype=float32

    def __init__(
        self,
        *,
        coefficient: float,
        distance: float,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.coefficient = tf.Variable(tf.constant(coefficient))
        self.distance = tf.Variable(tf.constant(distance))

    @tf.function(input_signature=(optical_signal_spec,))
    def _apply_pulse_dispersion(self, signal: OpticalSignal) -> OpticalSignal:
        wavelengths = Constants.c / signal.frequencies # m

        # Convert the dispersion coefficient from ps / (nm * km) to s / (m * m)
        D = self.coefficient * 1e-6  # s / m**2

        beta_2 = wavelengths ** 2 / (-2.0 * Constants.pi * Constants.c) * D

        fs = signal.timing.sampling_rate

        samples = signal.samples

        fft = tf.signal.fft(tf.transpose(samples))  # shape=(F, T)
        freqs = 2 * Constants.pi * tf.abs(fftfreq(tf.shape(fft)[1], 1.0 / fs))  # shape=(T,)

        phase_shifts = tf.reshape(
            tf.repeat(beta_2, tf.size(freqs)),
            [tf.size(beta_2), tf.size(freqs)]
        ) * 0.5 * self.distance * 1e3 * freqs ** 2

        new_signal = tf.transpose(
            tf.signal.ifft(
                tf.multiply(
                    fft,
                    tf.math.pow(
                        tf.cast(Constants.e, tf.complex64),
                        1j * tf.cast(phase_shifts, tf.complex64)  # type: ignore
                    )
                )
            )
        )

        return OpticalSignal(
            timing=signal.timing,
            frequencies=signal.frequencies,
            samples=new_signal,
            central_frequency=signal.central_frequency,
            power=signal.power
        )

    @tf.function(input_signature=(optical_signal_spec,))
    def _apply_linewidth_dispersion(self, signal: OpticalSignal) -> OpticalSignal:
        wavelength_deltas = (Constants.c * 1e9 / signal.frequencies) - (Constants.c * 1e9 / signal.central_frequency)

        amount = self.coefficient * self.distance

        # Each wavelength (or frequency) will be shifted in time by some number
        # of samples
        timeshifts = tf.cast(
            amount * 1e-12 * signal.timing.sampling_rate * wavelength_deltas,
            tf.int32
        )

        min_shift = tf.reduce_min(timeshifts)
        max_shift = tf.reduce_max(timeshifts)

        # We have to pad samples from both sides
        left_pad = tf.math.maximum(max_shift, 0)
        right_pad = tf.math.maximum(-min_shift, 0)

        padded_samples = tf.concat([
            tf.zeros(shape=[left_pad, tf.shape(signal.samples)[1]], dtype=tf.complex64),
            signal.samples,
            tf.zeros(shape=[right_pad, tf.shape(signal.samples)[1]], dtype=tf.complex64)
        ], 0)

        new_signal = tf.map_fn(
            lambda i: tf.map_fn(
                lambda j: padded_samples[left_pad + i - timeshifts[j], j],
                tf.range(tf.shape(signal.samples)[1]),
                fn_output_signature=tf.complex64
            ),
            tf.range(tf.shape(signal.samples)[0]),
            fn_output_signature=tf.complex64
        )

        return OpticalSignal(
            timing=signal.timing,
            frequencies=signal.frequencies,
            samples=new_signal,
            central_frequency=signal.central_frequency,
            power=signal.power
        )

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
        if tf.abs(self.coefficient) <= 1e-6:  # type: ignore
            return signal

        signal = self._apply_pulse_dispersion(signal)
        signal = self._apply_linewidth_dispersion(signal)

        return signal


class OpticalFiber(tf.Module):
    def __init__(
        self,
        attenuation: float,
        chromatic_dispersion: float,
        length: float,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self._optical_loss = OpticalLoss(loss=attenuation * length)
        self._chromatic_dispersion = ChromaticDispersion(coefficient=chromatic_dispersion, distance=length)

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
        return self._chromatic_dispersion(self._optical_loss(signal))


@tf.function(input_signature=(
    tf.TensorSpec(shape=[], dtype=tf.int32),
    tf.TensorSpec(shape=[], dtype=tf.float32)
))
def fftfreq(n: tf.Tensor, d: tf.Tensor) -> tf.Tensor:
    if tf.math.floormod(n, 2) == 0:  # type: ignore
        return tf.cast(
            tf.concat([
                tf.range(tf.cast(tf.math.floordiv(n, 2), tf.int32)),
                tf.range(tf.cast(-tf.math.floordiv(n + 1, 2), tf.int32), 0)
            ], 0),
            tf.float32
        ) / (d * tf.cast(n, tf.float32))
    else:
        return tf.cast(
            tf.concat([
                tf.range(tf.cast(tf.math.floordiv(n + 1, 2), tf.int32)),
                tf.range(tf.cast(-tf.math.floordiv(n - 1, 2), tf.int32), 0)
            ], 0),
            tf.float32
        ) / (d * tf.cast(n, tf.float32))


class ElectricalLowPassFilter(tf.Module):
    #: Filter cutoff frequency in Hz
    cutoff_frequency: tf.Variable  # shape=(1,); dtype=float32

    def __init__(self, cutoff_frequency: float, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.cutoff_frequency = tf.Variable(tf.constant(cutoff_frequency))

    @tf.function(input_signature=(tf.TensorSpec(shape=[2], dtype=tf.complex64),))
    def _transfer_function(self, fft_and_freq: tf.Tensor) -> tf.Tensor:
        fft = fft_and_freq[0]
        freq = tf.cast(tf.abs(fft_and_freq[1]), tf.float32)

        if tf.greater(freq, self.cutoff_frequency):  # type: ignore
            return tf.cast(tf.constant(0.0), tf.complex64)
        else:
            return fft

    @tf.function(input_signature=(electrical_signal_spec,))
    def __call__(self, signal: ElectricalSignal) -> ElectricalSignal:
        fft = tf.signal.fft(tf.cast(signal.samples, tf.complex64))
        freqs = fftfreq(tf.size(fft), 1.0 / signal.timing.sampling_rate)

        filtered_fft = tf.vectorized_map(
            fn=self._transfer_function,
            elems=tf.transpose(tf.stack([fft, tf.cast(freqs, tf.complex64)]))
        )

        filtered_samples = tf.signal.ifft(filtered_fft)

        return ElectricalSignal(
            timing=signal.timing,
            samples=filtered_samples
        )


class Photodiode(tf.Module):
    #: Minimum power (in watts) that can be converted into a current
    sensitivity: tf.Variable  # shape=(1,); dtype=float32

    #: Maximum power the diode can convert (anything above will be clipped)
    maximum_power: tf.Variable  # shape=(1,); dtype=float32

    def __init__(
        self,
        sensitivity: float = 1e-5,
        maximum_power: float = 2e-3,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sensitivity = tf.Variable(tf.constant(sensitivity))
        self.maximum_power = tf.Variable(tf.constant(maximum_power))

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, signal: OpticalSignal) -> ElectricalSignal:
        samples = tf.cast(
            tf.clip_by_value(
                tf.reduce_sum(tf.abs(signal.samples)**2, axis=1),
                self.sensitivity,
                self.maximum_power
            ),
            tf.complex64
        )

        return ElectricalSignal(
            timing=signal.timing,
            samples=samples
        )


class PAMDecoder(tf.Module):
    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Constellation used for encoding
    constellation: PAMConstellation

    #: Modulation order (number bits per symbol)
    order: tf.Variable  # shape=(1,); dtype=int32

    #: Minimum current (or power in watts) recognized as non-zero
    threshold: tf.Variable  # shape=(1,); dtype=float32

    #: Currents above this value will be clipped
    max_current: tf.Variable  # shape=(1,); dtype=float32

    def __init__(
        self,
        data_rate: float,
        order: int = 1,
        constellation: Optional[PAMConstellation] = None,
        threshold: float = 1e-5,
        max_current: float = 2e-3,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        if constellation is None:
            constellation = PAMConstellation(order=order)

        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.order = tf.Variable(tf.constant(order))
        self.constellation = constellation
        self.threshold = tf.Variable(tf.constant(threshold))
        self.max_current = tf.Variable(tf.constant(max_current))

    @tf.function(input_signature=(electrical_signal_spec,))
    def __call__(self, signal: ElectricalSignal) -> tf.Tensor:
        n = tf.cast(
            tf.round(signal.timing.sampling_rate / self.data_rate * tf.cast(self.order, tf.float32)),
            tf.int32
        )

        current = tf.clip_by_value(tf.abs(signal.samples), 0, self.max_current)
        max_current = tf.math.maximum(tf.reduce_max(current), self.threshold)

        symbols = tf.vectorized_map(
            fn=lambda i: tf.cast(
                tf.reduce_mean(current[i * n:(i + 1) * n]) / max_current,
                tf.complex64
            ),
            elems=tf.range(tf.cast(tf.size(current) / n, tf.int32))
        )

        return tf.reshape(
            tf.vectorized_map(
                fn=self.constellation.decode_symbol,
                elems=symbols
            ),
            (tf.size(symbols) * self.order,)
        )


class BERAnalyzer(tf.Module):
    @tf.function(input_signature=(
        bit_sequence_spec,
        bit_sequence_spec
    ))
    def __call__(self, expected_output: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
        tf.debugging.assert_equal(tf.size(output), tf.size(expected_output))

        correct_count = tf.cast(tf.math.count_nonzero(expected_output == output), tf.float32)
        total_bits = tf.cast(tf.size(output), tf.float32)

        return (total_bits - correct_count) / total_bits


class BERTester(tf.Module):
    #: System to test
    systems: list[Callable]

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        systems: Sequence[Callable],
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.systems = list(systems)

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

        self.ber_analyzer = BERAnalyzer()

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.int32)
    ))
    def __call__(self, sequence_length: tf.Tensor, n_iterations: tf.Tensor) -> tf.Tensor:
        inputs = tf.map_fn(
            fn=lambda i: self.rng.uniform(shape=(sequence_length,), minval=0, maxval=2, dtype=tf.int32),
            elems=tf.range(n_iterations)
        )

        bers = tf.zeros(shape=(0,), dtype=tf.float32)

        for system in self.systems:
            def fn(system):
                return lambda inp: self.ber_analyzer(inp, system(inp))

            bers = tf.concat(
                [
                    bers,
                    [tf.reduce_sum(tf.vectorized_map(fn(system), inputs))]
                ],
                0
            )

        return bers / tf.cast(n_iterations, tf.float32)


class System(tf.Module):
    #: System components
    components: list[Callable]

    def __init__(self, components: list[Callable], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.components = components

    @tf.function
    def __call__(self, inp: Any = None) -> Any:
        result: Any = inp

        for component in self.components:
            result = component(result)

        return result
