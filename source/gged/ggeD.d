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

auto gged(T,size_t N)(ulong[N] xyz)  
{
    return GgedStruct!(T*,N,false,mir_slice_kind.contiguous)(slice!(T)(xyz));
}
auto gged(T,Args...)(Args xyz)  if(allSameType!(Args) && isIntegral!(Args[0])) 
{
    return GgedStruct!(T*,Args.length,false,mir_slice_kind.contiguous)(slice!(T)(xyz));
}
auto gged(X,ulong Y,SliceKind Z)(Slice!(X,Y,Z) slice_) 
{
    return GgedStruct!(X,Y,false, Z)(slice_);
}
T gged(T)(T value) if(!__traits(isSame, TemplateOf!(T), Slice))
{
    return value;
}

private struct IndexLoop(GGED) 
{
    private alias MakeSerialIndex(X) =  SerialIndex!X;
    GGED gg;
    alias IndexTypes =GGED.IndexTypes;
    alias TypeSerialIndex = staticMap!(MakeSerialIndex,IndexTypes);

    int opApply(int delegate(TypeSerialIndex) fun) {
        mixin(genLoop!(false,0));
        return 1;
    }
    alias Rank = GGED.Rank;
    static if(Rank > 1)
    {
        int opApply(int delegate(IndexVec!(IndexTypes)) fun) {
            mixin(genLoop!(true,0));
            return 1;
        }
        static foreach(N ; 1 .. Rank -1)
        {
            int opApply(int delegate(IndexVec!(IndexTypes[0..Rank-N]),IndexTypes[Rank-N .. $]) fun) {
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

       	static if(vec) result ~= "IndexVec!(IndexTypes[0..Rank-"~N.to!string~"])(";
        static foreach(idx;0..Rank-N)
	        result ~= "SerialIndex!(IndexTypes["~idx.to!string~"])(gg.invIndex!"~idx.to!string~"(_" ~idx.to!string ~"),gg._slice.shape["~idx.to!string~"],gg.offsets["~idx.to!string~"]),";
       	static if(vec) result ~= "),";

        static foreach(idx;Rank - N .. Rank)
            result ~=  "SerialIndex!(IndexTypes["~idx.to!string~"])(gg.invIndex!"~idx.to!string~"(_" ~idx.to!string ~"),gg._slice.shape["~idx.to!string~"],gg.offsets["~idx.to!string~"]),";

        result ~= ");";
        return result;
    }
}

@nogc nothrow auto setOffset(GGED,ulong Rank)(GGED gged,double[Rank] offsets) if(is(GGED == GgedStruct!(T,Rank,isOffsetIndex,kind),T,bool isOffsetIndex,SliceKind kind))
{
    return GgedStruct!(GGED.TypePointer,GGED.Rank,true,GGED.Kind)(gged._slice,offsets.tupleof);
}

@nogc nothrow auto setOffset(R,ulong Rank)(R value,double[Rank] offsets) if(!is(R == GgedStruct!(T,Rank,isOffsetIndex,kind),T,bool isOffsetIndex,SliceKind kind) )
{
    return value;
}


@nogc nothrow auto setOffset(GGED,Args...)(GGED gged,Args offsets) 
{
    static if(is(GGED == GgedStruct!(T,Rank,isOffsetIndex,kind),T,ulong Rank,bool isOffsetIndex,SliceKind kind) && Args.length == Rank && allSatisfy!(isFloatingPoint,Args))
    return GgedStruct!(GGED.TypePointer,GGED.Rank,true,GGED.Kind)(gged._slice,offsets);
    else
    return gged;
}

@nogc auto sum(T,ulong N,bool offsetflag,SliceKind kind)(GgedStruct!(T,N,offsetflag,kind) g)
{
    auto result = 0;
    foreach(ijk;g.index)
    {
        result += g[ijk];
    }
    return result;
}

alias Gged(T,ulong Rank, bool isOffsetIndex = false) = GgedStruct!(T*,Rank,isOffsetIndex);

struct GgedStruct(T,ulong RANK, bool isOffsetIndex = false,SliceKind kind = SliceKind.contiguous)
{
	import mir.ndslice;
    alias SliceType = Slice!(T, Rank, kind);
    SliceType _slice;
    alias Kind = kind;

    static if(isOffsetIndex)
        alias IndexTypes = Repeat!(RANK,double);
    else
        alias IndexTypes = Repeat!(RANK,size_t);

    IndexTypes offsets = 0;

    private size_t getIndex(ulong n,Arg)(Arg idx)
    {
        return cast(size_t)(idx-offsets[n]);
    }

    auto dup()
    {
        auto d = gged!(Type)(shape).setOffset(offsets);
        foreach(ijk;this.index)
        {
            d[ijk] = this[ijk];
        }
        return d;
    }

    private IndexTypes[n] invIndex(ulong n)(size_t idx)
    {
        return cast(IndexTypes[n])(idx+offsets[n]);
    }
    private auto getIndexes(Args...)(Args idxs) if(Args.length == RANK)
    {
        template Filting(X)
        {
            static if(isContiguousSlice!X) alias Filting = X;
            else alias Filting = size_t;
        }
        struct Dummy{
            staticMap!(Filting,Args) value;
            alias value this;
        }
        Dummy result;
        static foreach(i;0..RANK)
        {
            static if(isContiguousSlice!(Args[i])) 
                result[i] =  idxs[i];
            else
                result[i] = getIndex!i(idxs[i]);
        }
        return result;
    }
    
    
    static if(isPointer!T)
    {
    	alias Type = PointerTarget!T;
    }
    else static if(isArray!T)
    {
        alias Type = ElementType!T;
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
    template PickupArray(Ns...)
        {
            alias PickupArray = AliasSeq!();
            static foreach(i; Ns)
            {
                PickupArray = AliasSeq!(PickupArray,offsets[i]);
            }
        }

    /// Ns次元だけを走査するindexでかえす
    @nogc nothrow auto index(Ns...)(){
        template allInDim(ulong N){alias allInDim= Alias!(N<Rank);}
        static assert(allSatisfy!(allInDim,Ns),"all Ns should be lower than Rank");
        auto newone = this.pick!(Ns)(offsets).setOffset(PickupArray!(Ns));
        return IndexLoop!(typeof(newone))(newone);
    }
    
    auto toString()=>_slice.to!string;

    /// pick!(Args...)(ijk)
    /// Argsにない次元の要素がijkの位置を含むArgs.length次元のgged配列を返す
    /// pick!0(ijk)では、ijkを含む0次元目への1次配列になる
    template pick(ns...)
    {
        @nogc nothrow auto pick(Args...)(Args value)
        {
            mixin(genPick([ns]));
        }
    }
        
    private static string genPick(ulong[] ns)
    {
        string result =  "return this[";
        foreach(i ; 0..Rank)
        {
            if(!ns.canFind(i))
                result ~= "value["~i.to!string~"],";
            else
                result ~= "offsets["~i.to!string~"] .. $,";
        }
        result ~= "];";
        return result;
    }
    @nogc nothrow auto opSlice(X,Y)(X start, Y end) if(RANK == 1 && is(X == SerialIndex!(IndexTypes[0])) && is(Y == SerialIndex!(IndexTypes[0])))
    {
        return gged(_slice.opSlice(start,end));
    }
    @nogc nothrow auto opSlice(size_t dim,X,Y)(X start, Y end) 
    {
        return _slice.opSlice!dim(getIndex!dim(start),getIndex!dim(end));
    }
    @nogc nothrow auto opIndex(IndexVec!(IndexTypes) args){
        return gged(_slice.opIndex(getIndexes(args.idx).value ));
    }
    static foreach(N; 1 .. Rank-1)
    {
        @nogc nothrow auto opIndex(Args...)(IndexVec!(IndexTypes[0..Rank-N]) args1,Args args2) if(Args.length == N) {
            return gged(_slice.opIndex(getIndexes(args1.idx,args2).value));
        }
    }
    @nogc nothrow auto opIndex(Args...)(Args args) if(Args.length == Rank) {
        return gged(_slice.opIndex(getIndexes(args).value));
    }
    import std.traits;
    static if( __traits(compiles, () {T itr; itr[0] = Type.init; } ))
    {
        @nogc nothrow auto opIndexAssign(AssignType)(AssignType value,IndexVec!(IndexTypes) args){
            _slice.opIndexAssign(value,getIndexes(args.idx).value);
        }
        
        static foreach(N; 1 .. Rank-1)
        {
            @nogc nothrow auto opIndexAssign(AssignType,Args...)(AssignType value,IndexVec!(IndexTypes[0..Rank-N])  args1,Args args2) if(Args.length == N) {
                _slice.opIndexAssign(value,getIndexes(args1.idx,args2).value);
            }
        }
        @nogc nothrow auto opIndexAssign(AssignType,Args...)(AssignType value,Args args) if(Args.length == Rank)  {
            _slice.opIndexAssign(value,getIndexes(args).value);
        }
    }
    @nogc nothrow auto opDollar(ulong dim)(){
        return invIndex!dim(_slice.opDollar!dim);
    }
    @nogc nothrow auto opDispatch(string idx)() if(idx.replace("_","").length == Rank)
    {
        return Leaf!("",idx.replace("_",""),typeof(this))(this);
    }
    auto opEquals(RHS)(RHS rhs)
    {
        static if(is(RHS == GgedStruct!(T2,RANK,isOffsetIndex2,kind2),T2,bool isOffsetIndex2,SliceKind kind2))
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

    auto opBinary(string op,GGED)(GGED rhs) 
    {
        static if(is(GGED==GgedStruct!(T2,RANK2,isOffsetIndex2,kind2),T2,bool isOffsetIndex2,SliceKind kind2))
            return _slice.opBinary!(op,T2,RANK2,kind2)(rhs._slice).gged.setOffset(offsets);
        else
            return _slice.opBinary!(op,GGED)(rhs).gged.setOffset(offsets);
    }
    
    auto opUnary(string op)()
        if (op == "*" || op == "~" || op == "-" || op == "+")
    {
        import mir.ndslice.topology: map;
        static if (op == "+")
            return this;
        else
            return _slice.opUnary!op.gged.setOffset(offsets);
    }
}