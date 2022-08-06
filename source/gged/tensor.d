/*
Copyright (c) 2022 Zenw
Released under the MIT license
https://opensource.org/licenses/mit-license.php
*/

module ggeD.tensor;

import ggeD.ggeD;
import std;
import std: canFind;

import ggeD.einsum;


/// make tensor from array
/// Params:
///   N = shape of tensor
auto tensor(T,X...)(X N) if(allSatisfy!(isIndex,X))
{
    return Tensor!(Gged!(T,X.length))(gged!(T,X)(N));
}

/// make tensor from array
/// Params:
///   value = sorce of array
///   N = shape of tensor
auto tensor(T,X...)(T[] value,X N) if(allSatisfy!(isIndex,X))
{
    return Tensor!(Gged!(T,X.length))(gged!(T,X)(value,N));
}

/// make tensor from gged array
/// Params:
///   value = sorce of array
///   N = shape of tensor
auto tensor(T,ulong Rank)(Gged!(T,Rank) gg)
{
    return Tensor!(Gged!(T,Rank))(gg);
}

/// 
/// Params:
///   gg = 
package(ggeD) auto tensor(T)(T gg) if(isBasicType!T)
{
    return gg;
}


struct Tensor(GG) if( __traits(isSame,TemplateOf!(GG) , Gged))
{
    GG _gged;
    private alias T = TemplateArgsOf!GG[0];
    private alias Rank = Alias!(TemplateArgsOf!GG[1]);
    alias _gged this;
    
    template opDispatch(string idx)
    {
        static if(__traits(hasMember,_gged,idx)) 
        {
            auto opDispatch()
            {
                return mixin("_gged."~idx);
            }
        }
        else
        {
            auto opDispatch() 
            {
                static assert(idx.length == Rank,"index length of tensor should be same with rank of the tensor;");
                return TensorIndexed!(to!(dchar[])(idx).filter!(a=>a!='_').to!string,"",GG)(_gged);
            }
        }
    }
    
    auto opUnary(string op)() if(op == "-")
    {
        auto gg = _gged.dup;
        foreach(a;gg)
        {
            mixin("gg[a] =  "~op~" gg[a];");
        }
        return tensor(gg);
    }
    auto opBinary(string op,R)(Tensor!(Gged!(R,Rank)) rhs) if((isOp(op[0]) || isOpPlusMinus(op)) && (is(myCommonType!(T,R) == T) || is(myCommonType!(T,R) == R )))
    {
        auto gg = Gged!(myCommonType!(T,R),Rank)(_gged.shape);
        foreach(a;gg)
        {
            mixin("gg[a] =  _gged[a] "~op~" rhs._gged[a];");
        }
        return tensor(gg);
    }
    auto opBinary(string op,R)( R rhs)  if(isNumeric!R)
    {
        auto gg = _gged.dup;
        foreach(a;gg)
        {
            mixin("gg[a] =  gg[a] "~op~" rhs;");
        }
        return tensor(gg);
    }
    auto opBinaryRight(string op, L)( L lhs)  if(isNumeric!L && op!="/")
    {
        auto gg = _gged.dup;
        foreach(a;gg)
        {
            mixin("gg[a] =  rhs "~op~" gg[a];");
        }
        return tensor(gg);
    }

    auto opIndex(X)(Vec!(Rank,X) arg) if(isIndex!X)
    {
        return _gged.opIndex!X(arg).tensor;
    }
    T opIndex(X...)(X arg) if(allSatisfy!(isIndex,X))
    {
        return _gged.opIndex!X(arg).tensor;
    }
    auto opIndex(X...)(X arg) if(!allSatisfy!(isIndex,X))
    {
        return _gged.opIndex!(Yes.Cut,X)(arg).tensor;
    } 
    
    void opIndexAssign(X)(T value,Vec!(Rank,X) arg) if(isIndex!X)
    {
        _gged.opIndex!X(arg) = value;
    }
    void opIndexAssign(X...)(T value,X arg) if(allSatisfy!(isIndex,X))
    {
        _gged.opIndex!X(arg) = value;
    }
    void opIndexAssign(X...)(T value,X arg) if(!allSatisfy!(isIndex,X))
    {
        _gged.opIndex!(Yes.Cut,X)(arg) = value;
    } 
    
    static if(_gged.RANK == 1)
    {
        auto opSlice(size_t dim,X,Y)(X start, Y end) if(isIndex!X && isIndex!Y)
        {
            return _gged.opSlice!(X,Y)(start,end).tensor;
        }
    }
    else
    {
        size_t[2] opSlice(size_t dim,X,Y)(X start, Y end) if(isIndex!X && isIndex!Y)
        {
            return [start, end];
        }
    }

    
    auto opDollar(ulong rank)()
    {
        return _gged.opDollar!(rank)();
    }
}
