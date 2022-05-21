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

struct TensorIterator<'a, Data>
{
    tensor: &'a Tensor<Data>,
    indices : Vec<usize>,
    offsets: Vec<isize>,
    index : isize,
    end : bool
}

fn get_linear_strides(extent: &[usize]) -> std::vec::Vec<isize> {
    let stride : Vec<isize> = [1].iter().chain(extent.iter().rev()).scan(1, |prod, e| {*prod *=  *e as isize; Some(*prod)}).collect();
    stride.iter().rev().skip(1).cloned().collect::<Vec<isize>>()
}

fn broadcast_extents(lhs: &[usize], rhs: &[usize]) -> std::vec::Vec<usize> {
    lhs.iter().zip(rhs.iter()).map(|(&l, &r)|{ if l==1 { r } else if r==1{ l } else { assert_eq!(l, r); l}}).collect::<Vec<usize>>()
}

impl<'a, Data> Iterator for TensorIterator<'a, Data> where Data: std::fmt::Debug+Copy
{
    type Item = Data;
    fn next(&mut self) -> std::option::Option<<Self as std::iter::Iterator>::Item>
    {
        if self.end {
            return None;
        }
        let cur_index = self.index;
        let mut carry = 1;
        let mut carry_idx = self.tensor.extent.len() as isize - 1;
        for (idx, ext) in self.indices.iter_mut().zip(self.tensor.extent.iter()).rev() {
            let next = *idx + carry;
            carry = next / *ext;
            *idx = next % *ext;
            if carry == 0 {
                break;
            }
            carry_idx = carry_idx - 1;
        }
        if carry_idx <= -1 {
            self.end = true;
        }
        else
        {
            self.index = self.index + self.offsets[carry_idx as usize]
        }
        return Some(self.tensor.index(cur_index))
    }
}

impl<'a, Data: std::fmt::Debug + Copy> IntoIterator for &'a Tensor<Data>
{
    type Item = Data;
    type IntoIter = TensorIterator<'a, Data>;
    fn into_iter(self) -> Self::IntoIter {
        let mut offsets = vec![0; self.extent.len()];
        for (i, offset) in offsets.iter_mut().enumerate() {
            *offset = (self.stride[i] - (if i < (self.extent.len() - 1) { (self.extent[i + 1] - 1) as isize * self.stride[i + 1] } else {0})) as isize;
        }
        Self::IntoIter{tensor: self, indices: vec![0; self.extent.len()], offsets : offsets, index : 0, end: false}
    }
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
    fn to_contiguous(&self) -> Self
    {
        let stride = get_linear_strides(&self.extent);
        let data : Vec<Data> = self.into_iter().collect();
        Tensor { extent : self.extent.clone(), stride : stride, data : Rc::new(data) , offset : 0}
    }
    fn index(&self, idx: isize) -> Data
    {
        self.data[(self.offset as isize + idx) as usize]
    }
    fn at(&self, g: &[usize]) -> Data
    {
        assert_eq!(g.len(), self.stride.len());
        for (&x, &y) in self.extent.iter().zip(g.iter()) {
            assert!(x>y);
        }
        let indices : isize = self.stride.iter().zip(g.iter()).map(|(&x, &y)|x* y as isize).sum();
        self.data[(self.offset as isize + indices) as usize]
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

fn unary<Data: Copy+Debug>(tensor : &Tensor<Data>, f :&dyn Fn(Data)->Data) -> Tensor<Data>
{
    let stride = get_linear_strides(&tensor.extent);
    let data = tensor.into_iter().map(f).collect();
    Tensor { extent : tensor.extent.clone(), stride : stride, data : Rc::new(data) , offset : 0}
}

fn tensor_for_broadcast<Data>(tensor : &Tensor<Data>, extent: &Vec<usize>) -> Tensor<Data>
{
    let stride = tensor.extent.iter().zip(tensor.stride.iter()).map(|(&x, & y)| if x == 1 { 0 } else {y}).collect();
    Tensor { extent : extent.clone(), stride : stride, data : tensor.data.clone(), offset : 0}
}

fn binary<Data: Copy+Debug>(lhs : &Tensor<Data>, rhs : &Tensor<Data>, f :&dyn Fn(Data, Data)->Data) -> Tensor<Data>
{
    let extent = broadcast_extents(&lhs.extent, &rhs.extent);
    let stride = get_linear_strides(&extent);
    let data = tensor_for_broadcast(lhs, &extent).into_iter().zip(tensor_for_broadcast(rhs, &extent).into_iter()).map(|(x, y)|f(x, y)).collect();
    Tensor { extent : extent, stride : stride, data : Rc::new(data) , offset : 0}
}

fn reshape<Data>(tensor : &Tensor<Data>, shape: &[usize]) -> Tensor<Data> where Data: std::fmt::Debug + Copy
{
    assert_eq!(shape.into_iter().product::<usize>(), (*(&tensor).extent).into_iter().product::<usize>());
    let mut involved_dims = Vec::new();
    let mut i = 0;
    let mut j = 0;
    let mut i_product = 1;
    let mut j_product = 1;
    while i < shape.len() && j < tensor.extent.len() {
        if i_product == 1 && j_product == 1 && shape[i] == tensor.extent[j]
        {
            i += 1;
            j += 1;
        }
        else
        {
            i_product *= shape[i];
            j_product *= shape[j];
            involved_dims.push(j);
            while i_product != j_product
            {
                if i_product < j_product
                {
                    i += 1;
                    i_product *= shape[i];
                }
                else
                {
                    j += 1;
                    j_product *= shape[j];
                    involved_dims.push(j);
                }
            }
            i_product = 1;
            j_product = 1;
            i += 1;
            j += 1;
        }
    }
    let noop : bool = involved_dims.iter().map(|&i| i + 1 == tensor.stride.len() || tensor.stride[i] == (tensor.extent[i] as isize * tensor.stride[i+1])).fold(true, |sum, e| sum && e);
    if noop {
        Tensor { extent : tensor.extent.clone(), stride : tensor.stride.clone(), data : tensor.data.clone(), offset : tensor.offset}
    }
    else {
        let stride = get_linear_strides(&shape);
        let data : Vec<Data> = tensor.into_iter().collect();
        Tensor { extent : shape.to_vec(), stride : stride, data : Rc::new(data) , offset : 0}
    }
}

fn main() {
    let x = Tensor::from2d(&[&[ 11, 12, 13, 14], &[21, 22, 23, 24], &[31, 32, 33, 34], &[41, 42, 43, 44]]);
    let y = slice(&x, &[1, 3], &[3,0], &[1, -1]);
    let z = permute(&y, &[1, 0]);
    let a = reshape(&y, &[3, 2, 1]);
    let u = unary(&a, &|x : i32|-x);
    let t = Tensor::from2d(&[&[ 11, 12, 13, 14]]);
    let zero = binary(&x, &t, &|x, y|x+y);
    for c in &[x, y, z ,a, u, zero] {
        println!("c\n{:?}", c);
        for i in c {
            print!("{}, ", i)
        }
        println!("{}", "")
    }
}
