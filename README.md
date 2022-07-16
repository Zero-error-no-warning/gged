# ggeD

A ~~ja~~gged array library that saves you from writing nested for statements in multidimensional arrays.

ジャグ配列じゃない多次元配列ライブラリです。入れ子にせずループを回すための機能を提供します。

``` D
import ggeD;
import std.range : iota;
import std.array : array;
import std.math : sin;
import std.stdio : writeln;
void main()
{
    auto A = gged!double(iota(27.).array,3,3,3); // create A gged array
    foreach(ijk ; A) // parallel foreach
    {
        A[ijk] = sin(A[ijk]); // easy to access
        A[ijk] = 1.*ijk[0] + 2.*ijk[1] + 3.*ijk[2]; // ijk is vector
    }
    foreach(i,j,k ; A.Serial) // explicit serial foreach
    {
        writeln(A[i,j,k]); //  also easy to access  
    }
}
```

And also included tensor library.
You can use Einstein summation.

アインシュタインの縮約記法が使えるテンソルライブラリもあります。

``` D

import ggeD : tensor , Einsum;
import std.range : iota;
import std.array : array;
import std.stdio : writeln;

void main()
{
	auto A = tensor!double(iota(9.).array,3,3); // 2 rank tensor(matrix)
    auto B = tensor!double([3.,2,1],3); // 1 rank tensor(vector)
    auto C = Einsum | A.ij*B.i; // You can write an expressions without for statement.
    foreach(ijk ; C.Serial) // You can use as gged array;
    {
        writeln(ijk, " | " ,C[ijk]); 
    }

    auto tr = Einsum | A.ii; // trace of A.
    assert

    import ggeD : BroadCast; // BroadCast makes a function usable in Einstein summation.
    import std.math : sin; 
    auto D  = Einsum | A.ij + BroadCast!sin(B.i*B.j); 
}

```