/*
Copyright (c) 2022 Zenw
Released under the MIT license
https://opensource.org/licenses/mit-license.php
*/

module ggeD.ggeD;
import std;
public import ggeD.indexVec;
import mir.ndslice;

public import ggeD.einsum;



auto gged(T,Args...)(T[] value,Args xyz)
{
    auto gg = gged!T(xyz);
    ulong idx = 0;
    foreach(ijk;gg.index)
    {
        gg[ijk] = value[idx];
        idx++;
    }
    return gg;
}
auto gged(T,N)(T[] value,ulong[N] xyz)  
{
    auto gg = gged!T(xyz);
    ulong idx = 0;
    foreach(ijk;gg.index)
    {
        gg[ijk] = value[idx];
        idx++;
    }
    return gg;
}

auto gged(T,N)(ulong[N] xyz)  
{
    return Gged!(T*,Args.length,mir_slice_kind.contiguous)(slice!(T)(xyz));
}
auto gged(T,Args...)(Args xyz)  if(allSameType!(Args) && isIntegral!(Args[0]))
{
    return Gged!(T*,Args.length,mir_slice_kind.contiguous)(slice!(T)(xyz));
}
auto gged(X,ulong Y,SliceKind Z)(Slice!(X,Y,Z) slice_)
{
    return Gged!(X,Y, Z)(slice_);
}
T gged(T)(T value) if(!__traits(isSame, TemplateOf!(T), Slice))
{
    return value;
}


struct Gged(T,ulong RANK, SliceKind kind)
{
	import mir.ndslice;
    alias SliceType = Slice!(T, Rank, kind);
    SliceType _slice;
    alias Kind = kind;
    
	alias Type = PointerTarget!T;
	alias TypePointer = T;
	alias Rank = Alias!(RANK);

    alias _slice this;
    
    auto shape() => _slice.shape;
    
    alias TypeSerialIndex = Repeat!(Rank,SerialIndex);
    auto index(){
        return new class{
            	int opApply(int delegate(TypeSerialIndex) fun) {
                	mixin(genLoop);
            	    return 1;
    	    }
            	static if(Rank > 1)
                    int opApply(int delegate(IndexVec!Rank) fun) {
                	mixin(genLoop!true);
            	    return 1;
    	    }
        };
    }
    auto toString()=>_slice.to!string;
    static string genLoop(bool vec = false)(){
        string result;
        static foreach(idx;0..Rank){
            result ~= "foreach(_" ~idx.to!string~ ";0 .. _slice.shape["~idx.to!string~"])";
        }
        result ~= "fun(";
       	static if(vec) result ~= "IndexVec!Rank([";
        static foreach(idx;0..Rank){
	        result ~= "SerialIndex(_" ~idx.to!string ~",_slice.shape["~idx.to!string~"]),";
        }
       	static if(vec) result ~= "])";
        result ~= ");";
        return result;
    }
    auto opSlice(X,Y)(X start, Y end) if(is(X == SerialIndex) && is(Y == SerialIndex))
    {
        return gged(_slice.opSlice(start,end));
    }
    auto opSlice(size_t dim,X,Y)(X start, Y end) 
    {
        return _slice.opSlice!dim(start,end);
    }
    auto opIndex(IndexVec!Rank args){
        return gged(_slice.opIndex(args.idx.tupleof));
    }
    auto opIndex(Args...)(Args args){
        return gged(_slice.opIndex(args));
    }
    auto opIndexAssign(PointerTarget!T value,IndexVec!Rank args){
        return gged(_slice.opIndexAssign(value,args.idx.tupleof));
    }
    auto opIndexAssign(Args...)(PointerTarget!T value,Args args){
        return gged(_slice.opIndexAssign(value,args));
    }
    auto opDollar(ulong dim)(){
        return _slice.opDollar!dim;
    }
    auto opDispatch(string idx)()
    {
        static assert(idx.length ==Rank,"index notation length should be same as the tensor Rank");
        return new Leaf!(idx,typeof(this))(this);
    }
    auto opEquals(RHS)(RHS rhs)
    {
        bool result = true;
        static if(is(RHS == Gged!(T2,RANK,kind2),T2,kind2))
        {
            foreach(ijk ; index)
            {
                result &= rhs[ijk] == this[ijk];
            }
        }
        else
        {
            _slice == rhs;
        }
        return result;
    }
}
