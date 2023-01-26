# ggeD

A ~~ja~~gged array library that saves you from writing nested for statements in multidimensional arrays.

ジャグ配列じゃない多次元配列ライブラリです。入れ子にせずループを回すための機能を提供します。

``` D
import ggeD;
import std;
void main()
{
    auto A = iota(27.).array.gged!double(3,3,3); // create a gged array
    foreach(ijk ; A.index) 
    {
        A[ijk] = sin(A[ijk]); // easy to access
        A[ijk] = 1.*ijk[0] + 2.*ijk[1] + 3.*ijk[2]; // ijk is like vector
    }
    foreach(i,j,k ; A.index)
    {
        writeln(A[i,j,k]); //  also easy to access  
    }
}
```

You can use Einstein summation.

アインシュタインの縮約記法が使えます。

``` D
import ggeD;
import std;

void main()
{
    auto t1 = iota(9.).array.gged!double(3,3);
    assert(t1 == [[0, 1, 2],[3, 4, 5],[6, 7, 8]]);
    
    auto tr = Einsum | t1.ii;
    assert(tr == 12);

    auto transposed = Einsum.ji | t1.ij;
    assert(transposed == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]);

    auto delta = fnTensor((ulong i,ulong j)=>(i==j?1.:0.));
    auto tr2 = Einsum | t1.ij*delta.ij;
    assert(tr2 == 12);

    auto applyFunction = Einsum | br!tan(br!atan(t1.ij));
    assert(t1 == applyFunction);

    auto applyFunction2 = Einsum | br!atan2(t1[0,0..3].i,1.+t1[0..3,0].i);
    assert(applyFunction2 == atan2(t1[0,0],1+t1[0,0]) + atan2(t1[0,1],1+t1[1,0]) + atan2(t1[0,2],1+t1[2,0]) );
}

```