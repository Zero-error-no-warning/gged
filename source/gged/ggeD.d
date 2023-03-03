/*
Copyright (c) 2022 Zenw
Released under the MIT license
https://opensource.org/licenses/mit-license.php
*/

module ggeD.ggeD;
import std;
import ggeD.indexVec;
import mir.ndslice;
import ggeD.einsum;
import mir.ndslice.topology : iota;


auto gged(T,R,Args...)(R value,Args xyz) if(isInputRange!(Unqual!R))
{
    auto gg = gged!T(xyz);
    ulong idx = 0;
    foreach(ijk;gg.index)
    {
        gg[ijk] = value.front;
        value.popFront();
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

private struct IndexLoop(GGED) 
{
    GGED gg;
    alias TypeSerialIndex = Repeat!(Rank,SerialIndex);
    int opApply(int delegate(TypeSerialIndex) fun) {
        mixin(genLoop!(false,0));
        return 1;
    }
    alias Rank = GGED.Rank;
    static if(Rank > 1)
    {
        int opApply(int delegate(IndexVec!(Rank)) fun) {
            mixin(genLoop!(true,0));
            return 1;
        }
        static foreach(N ; 1 .. Rank -1)
        {
            int opApply(int delegate(IndexVec!(Rank-N),Repeat!(N,SerialIndex)) fun) {
                mixin(genLoop!(true,N));
                return 1;
            }
        }
    }
    static string genLoop(bool vec,ulong N)(){
        string result;
        static foreach(idx;0..Rank){
            result ~= "foreach(_" ~idx.to!string~ ";0 .. gg._slice.shape["~idx.to!string~"])\n";
        }
        result ~= "fun(";

       	static if(vec) result ~= "IndexVec!(Rank-"~N.to!string ~")([";
        static foreach(idx;0..Rank-N)
	        result ~= "SerialIndex(_" ~idx.to!string ~",gg._slice.shape["~idx.to!string~"]),";
       	static if(vec) result ~= "]),";

        static foreach(idx;Rank - N .. Rank)
            result ~=  "SerialIndex(_" ~idx.to!string ~",gg._slice.shape["~idx.to!string~"]),";

        result ~= ");";
        return result;
    }
}

private string genPick(ulong n,string value,ulong Rank)
{
    string result =  "return this[";
    foreach(i ; 0..Rank)
    {
        if(i == n )
            result ~= value~",";
        else
            result ~= "0 .. $,";
    }
    result ~= "];";
    return result;
}

struct Gged(T,ulong RANK, SliceKind kind)
{
	import mir.ndslice;
    alias SliceType = Slice!(T, Rank, kind);
    SliceType _slice;
    alias Kind = kind;
    
    static if(isPointer!T)
    {
    	alias Type = PointerTarget!T;
    }
    else
    {
        alias Type = ReturnType!(T.opUnary!"*");
    } 
	alias TypePointer = T;
	alias Rank = Alias!(RANK);

    alias _slice this;
    
    @nogc nothrow auto shape() => _slice.shape; 
    
    @nogc nothrow auto index(){
        return IndexLoop!(typeof(this))(this);
    }
    auto toString()=>_slice.to!string;

    @nogc nothrow auto pick(ulong n)(ulong value)
    {
        mixin(genPick(n,"value",Rank));
    }
    @nogc nothrow auto opSlice(X,Y)(X start, Y end) if(is(X == SerialIndex) && is(Y == SerialIndex))
    {
        return gged(_slice.opSlice(start,end));
    }
    @nogc nothrow auto opSlice(size_t dim,X,Y)(X start, Y end) 
    {
        return _slice.opSlice!dim(start,end);
    }
    @nogc nothrow auto opIndex(IndexVec!Rank args){
        return gged(_slice.opIndex(args.idx.tupleof));
    }
    static foreach(N; 1 .. Rank-1)
    {
        @nogc nothrow auto opIndex(Args...)(IndexVec!(Rank-N) args1,Args args2) if(Args.length == N) {
            return gged(_slice.opIndex(args1.idx.tupleof,args2));
        }
    }
    @nogc nothrow auto opIndex(Args...)(Args args) if(Args.length == Rank) {
        return gged(_slice.opIndex(args));
    }
    import std.traits;
    static if( __traits(compiles, () {T itr; itr[0] = Type.init; } ))
    {
        @nogc nothrow auto opIndexAssign(Type value,IndexVec!Rank args){
            return gged(_slice.opIndexAssign(value,args.idx.tupleof));
        }
        
        static foreach(N; 1 .. Rank-1)
        {
            @nogc nothrow auto opIndexAssign(Args...)(Type value,IndexVec!(Rank-N) args1,Args args2) if(Args.length == N) {
                return gged(_slice.opIndex(value,args1.idx.tupleof,args2));
            }
        }
        @nogc nothrow auto opIndexAssign(Args...)(Type value,Args args) if(Args.length == Rank)  {
            return gged(_slice.opIndexAssign(value,args));
        }
    }
    @nogc nothrow auto opDollar(ulong dim)(){
        return _slice.opDollar!dim;
    }
    @nogc nothrow auto opDispatch(string idx)()
    {
        static assert(idx.length ==Rank,"index notation length should be same as the tensor Rank");
        return Leaf!("",idx,typeof(this))(this);
    }
    auto opEquals(RHS)(RHS rhs)
    {
        static if(is(RHS == Gged!(T2,RANK,kind2),T2,SliceKind kind2))
        {
            bool result = true;
            foreach(ijk ; index)
            {
                result &= isClose(rhs[ijk] , this[ijk] );
            }
            return result;
        }
        else
        {
            return _slice == rhs;
        }
    }
}
