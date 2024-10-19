use std::time::Duration;

use peony::signal::Signal;
use rodio::{OutputStream, Sink};

fn main() {
    let signal = Signal::tone(440.0, 44100, Duration::from_secs(3));

    let autocorr = signal.autocorrelate(None);

    println!("Rezultat autocorelare:");
    for channel in autocorr.iter() {
        for sample in channel.iter().take(5) {
            print!("{:.4} ", sample);
        }
        println!();
    }
    println!();

    let coeffs = signal.lpc(16);

    println!("Coefficienti de predictie liniara:");
    for channel in coeffs.iter() {
        for coeff in channel.iter().take(5) {
            print!("{:.4} ", coeff);
        }
        println!();
    }
    println!();

    println!("Inainte de cuantizare");
    for channel in signal.samples.iter() {
        for coeff in channel.iter().skip(10000).take(5) {
            print!("{:.4} ", coeff);
        }
        println!();
    }
    println!();

    let signal = signal.mu_quantize::<u8>(255);

    println!("Dupa cuantizare");
    for channel in signal.samples.iter() {
        for coeff in channel.iter().skip(10000).take(5) {
            print!("{:.4} ", coeff);
        }
        println!();
    }
    println!();

    let signal = signal.mu_dequantize::<f32>(255);

    println!("Dupa decuantizare");
    for channel in signal.samples.iter() {
        for coeff in channel.iter().skip(10000).take(5) {
            print!("{:.4} ", coeff);
        }
        println!();
    }
    println!();

    // let source = signal.into_rodio_source::<f32>();

    // let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    // let sink = Sink::try_new(&stream_handle).unwrap();
    // sink.append(source);
    // sink.sleep_until_end()
}
