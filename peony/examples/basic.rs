use peony::signal::{ResampleType, Signal};
use rodio::{OutputStream, Sink};

fn main() {
    let signal: Signal<f32> = Signal::load_default("Shape of You.wav").unwrap();

    println!("Number of channels: {}", signal.channels());
    println!();

    println!("Sample rate: {}", signal.sample_rate);
    println!();

    println!("Song duration: {:?}", signal.duration());
    println!();

    println!("Esantioane:");
    for channel in signal.samples.iter() {
        for sample in channel.iter().skip(1000).take(5) {
            print!("{:.4} ", sample);
        }
        println!();
    }
    println!();

    let signal = signal.into_mono();
    let signal = signal.into_resample(60000, ResampleType::Fft).unwrap();

    let signal: Signal<f32> = Signal::load_default("Shape of You.wav")
        .unwrap()
        .into_mono()
        .into_resample(100000, ResampleType::SincHighQuality)
        .unwrap();

    let source = signal.into_rodio_source::<f32>();

    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    let sink = Sink::try_new(&stream_handle).unwrap();
    sink.append(source);
    sink.sleep_until_end()
}
