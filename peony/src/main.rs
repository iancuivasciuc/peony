pub fn test<T>(x: i32) {
    println!("{}", std::any::type_name::<T>());
}

fn main() {
    let x = 5;

    test(x);
}
