use std::fmt::Debug;
use std::rc::Rc;

#[derive(Debug)]
struct Tensor<Data>
{
    extent : Vec<usize>,
    stride : Vec<isize>,
    data : Rc<Vec<Data>>,
    offset : usize
}

impl<Data> Tensor<Data> where Data: Debug + Copy
{
    fn new() -> Self
    {
        Tensor { extent : Vec::new(), stride : Vec::new(), data : Rc::new(Vec::new()), offset : 0}
    }
    fn nums(value: Data, num: usize) -> Self
    {
        Tensor { extent : vec![num; 1], stride : vec![1; 1], data : Rc::new(vec![value; num]), offset : 0}
    }
    fn single(value: Data) -> Self
    {
        Self::nums(value, 1)
    }
    fn from(data: &[Data]) -> Self
    {
        Tensor { extent : vec![data.len(); 1], stride : vec![1; 1], data : Rc::new(data.to_vec()), offset : 0}
    }
    fn from2d(data: &[&[Data]]) -> Self
    {
        let mut d_ = Vec::<Data>::new();
        for d in data {
            for e in *d {
                d_.push(*e);
            }
        }
        Tensor { extent : [data.len(), data[0].len()].to_vec(), stride : [data[0].len() as isize, 1].to_vec(), data : Rc::new(d_) , offset : 0}
    }
    fn at(&self, indices: &[usize]) -> Data
    {
        assert_eq!(indices.len(), self.stride.len());
        for (&x, &y) in self.extent.iter().zip(indices.iter()) {
            assert!(x>y);
        }
        let index : isize = self.stride.iter().zip(indices.iter()).map(|(&x, &y)|x* y as isize).sum();
        self.data[(self.offset as isize + index) as usize]
    }
}

fn slice<Data>(tensor: &Tensor<Data>, begins: &[usize], ends: &[usize], steps: &[isize]) -> Tensor<Data>
{
    assert_eq!(begins.len(), tensor.stride.len());
    assert_eq!(begins.len(), ends.len());
    assert_eq!(begins.len(), steps.len());
    for (&x, &y) in tensor.extent.iter().zip(begins.iter()) {
        assert!(x>y);
    }
    for (&x, &y) in tensor.extent.iter().zip(ends.iter()) {
        assert!(x>y);
    }
    let offset : usize = tensor.stride.iter().zip(begins.iter()).map(|(&x, &y)|x as usize *y).sum();
    let extent : Vec<usize> = begins.iter().zip(ends.iter()).zip(steps.iter()).map(|((&x, &y), &z)|(((y as isize)- (x as isize))/z) as usize).collect();
    let stride = tensor.stride.iter().zip(steps.iter()).map(|(&x, &y)|x*y).collect();
    Tensor { extent : extent, stride : stride, data : tensor.data.clone(), offset : offset}
}

fn permute<Data>(tensor : &Tensor<Data>, permutation: &[usize]) -> Tensor<Data>
{
    let len = permutation.len();
    assert_eq!(len, tensor.extent.len());
    for p in permutation {
        assert!(*p < len);
    }
    let mut extent = vec![0; len];
    let mut stride = vec![0; len];
    for (i, p) in permutation.iter().enumerate() {
        extent[i] = tensor.extent[*p];
        stride[i] = tensor.stride[*p];
    }
    Tensor { extent : extent, stride : stride, data : tensor.data.clone(), offset : tensor.offset}
}

fn main() {
    let x = Tensor::from2d(&[&[ 4.0, 3.0, 6.0], &[3.2, 2.0, 4.0], &[6.0, 1.5, 6.7]]);
    println!("{}", x.at(&[1, 0]));
    let y = slice(&x, &[1, 2], &[2,0], &[1, -1]);
    println!("{:?}", y);
    println!("{}", y.at(&[0, 0]));
    println!("{}", y.at(&[0, 1]));
    let z = permute(&y, &[1, 0]);
    println!("{:?}", z);
    println!("{}", z.at(&[0, 0]));
    println!("{}", z.at(&[1, 0]));
}
