/*
Copyright (c) 2022 Zenw
Released under the MIT license
https://opensource.org/licenses/mit-license.php
*/
module ggeD.einsum;

import std;
import ggeD.ggeD;


unittest
{
   auto A = iota(9.).gged!double(3,3);
    auto x = A[0,0..$];
    assert(A == [[0, 1, 2],[3, 4, 5],[6, 7, 8]]);
    assert(x == [0, 1, 2]);

    auto Ax = Einsum | A.ij * x.i;
    assert(Ax == [15,18,21]);

    auto tr = Einsum | A.ii;
    assert(tr == 12);

    auto transposed = Einsum.ji | A.ij;
    assert(transposed == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]);

    auto delta = fnTensor((ulong i,ulong j)=>(i==j?1.:0.));
    auto tr2 = Einsum | A.ij*delta.ij;
    assert(tr2 == 12);

    auto applyFunction = Einsum | br!tan(br!atan(A.ij)*1.);
    assert(A == applyFunction);

    auto applyFunction2 = Einsum | br!atan2(A[0,0..3].i,1.+A[0..3,0].i);
    assert(applyFunction2 == atan2(A[0,0],1+A[0,0]) + atan2(A[0,1],1+A[1,0]) + atan2(A[0,2],1+A[2,0]) );
}


@nogc auto br(alias fun,Args...)(Args args)
{
    return Func!("",fun,Args)(args);
}



@nogc auto fnTensor(F)(F fun) if(isCallable!F)
{
    struct Sub
    {
        auto opDispatch(string idx)() 
        {
            return FnTensor!(idx,F)(fun);
        }
    }
    return Sub();
}


struct Einsum
{
    static:
    auto evalnogc(string ResultIdx="",Node)(Node leaf) @nogc nothrow
    {
        import mir.ndslice;
        import ggeD.iterator;
        alias indexes  = getIndex!(Node,ResultIdx);
        alias Origin = TemplateOf!Node;
        alias Args = TemplateArgsOf!Node;
        alias NewOne =  Origin!(ResultIdx,Args[1..$]);
        auto newone = NewOne(leaf.tupleof);
        static if(indexes[0].length > 0)
        {
            alias shapes= Alias!(getShapes!(NewOne,"newone",indexes[0]));
            auto arr =  mixin("iota("~shapes~")");
            size_t[indexes[0].length] shape = mixin("["~shapes~"]");
            auto itr = EinsumIterator!(typeof(arr._iterator),typeof(arr),NewOne)(arr._iterator,arr,newone);
            return Slice!(typeof(itr),indexes[0].length)(shape,itr).gged;
        }
        else
        {
            return newone.calc();
        }
    }

    auto opBinary(string op,R)(R rhs) @nogc nothrow  if(op == "|") 
    {
        return evalnogc(rhs);
    }
    
    auto opDispatch(string name)()  @nogc nothrow 
    { 
        struct Sub
        {
            auto opBinary(string op,R)(R rhs) if(op == "|")
            {
                return evalnogc!name(rhs);
            }
        }

        return Sub();
    }
   
}

package(ggeD):
string onlyUniq(string input,string ignr="")
{
    return ignr != "" ? ignr : input.to!(dchar[]).filter!(a=>input.count(a) == 1).array.to!string;
}
string onlyDummy(string input,string ignr="")
{
    return ignr != "" ? input.to!(dchar[]).filter!(a=>ignr.count(a)==0).array.sort.uniq.array.to!string :  input.to!(dchar[]).filter!(a=>input.count(a) > 1 ).array.sort.uniq.array.to!string;
}


template filterTensors(Leafs...)
{
    static if(Leafs.length == 1)
    {
        static if(is(Leafs[0] == Leaf!(Ridx,idx,T),string Ridx,string idx,T) || isBasicType!(Leafs[0]))
        {
            alias filterTensors = Leafs[0];
        }
        else static if(is(Leafs[0] == Func!(Ridx,func,Leafs_),string Ridx,alias func,Leafs_...))
        {
            alias filterTensors = filterTensors!(Leafs_);
        }
        else static if(is(Leafs[0] == Tree!(Ridx,L,R,OP,Leafs_),string Ridx,L,R,string OP,Leafs_...))
        {
            alias filterTensors = filterTensors!(Leafs_);
        }
        else 
        {
            alias filterTensors = AliasSeq!();
        }
    }
    else static if(Leafs.length == 0)
    {
        alias filterTensors = AliasSeq!();
    }
    else
    {
        alias filterTensors = AliasSeq!(filterTensors!(Leafs[0]),filterTensors!(Leafs[1..$]));
    }
}
string removeCharacters(string A, string B)
{
    // 文字列Bに含まれる文字をフィルタリングして削除する
    return A.filter!(c => !B.canFind(c)).array.to!string;
}

template getIndex(Node,string ignr = "")
{
    template getunq(Leaf)
    {
        alias getunq = Alias!(getIndex!(Leaf,ignr)[0]);
    }
    template getdmy(Leaf)
    {
        alias getdmy = Alias!(getIndex!(Leaf,ignr)[1]);
    }
    static if(is(Node == Tree!(Ridx,L,R,OP,Leafs),string Ridx,L,R,string OP,Leafs...))
    {
        alias LHS = getIndex!(L,ignr);
        alias RHS = getIndex!(R,ignr);
        static if(OP == "*" || OP == "/")
        {
            // alias getIndex = AliasSeq!((LHS[0]~RHS[0]~LHS[1]~RHS[1]).onlyUniq(ignr),((LHS[0]~RHS[0]).onlyDummy(ignr)~LHS[1]~RHS[1]).array.sort.uniq.array.to!string);
            alias getIndex = AliasSeq!((LHS[0]~RHS[0]~LHS[1]~RHS[1]).onlyUniq(ignr).removeCharacters(LHS[1]~RHS[1]),((LHS[0]~RHS[0]).onlyDummy(ignr)~LHS[1]~RHS[1]).array.sort.uniq.array.to!string);
            // alias getIndex = AliasSeq!((LHS[0]~RHS[0]~LHS[1]~RHS[1]).onlyUniq(ignr),(LHS[0]~RHS[0]~LHS[1]~RHS[1]).onlyDummy(ignr));
        }
        else 
        {
            alias getIndex = AliasSeq!((LHS[0]~RHS[0]).array.sort.uniq.array.to!string ,"");
            // alias getIndex = AliasSeq!((LHS[0]~RHS[0]).onlyUniq(ignr),((LHS[0]~RHS[0]).onlyDummy(ignr)~LHS[1]~RHS[1]).array.sort.uniq.array.to!string);
        }
    }
    else static if(is(Node == Leaf!(Ridx,idx,Ts),string Ridx,string idx,Ts))
    {
        alias getIndex = AliasSeq!(idx.onlyUniq(ignr),idx.onlyDummy(ignr));
    }
    else static if(is(Node == Func!(Ridx,fun,Leafs),string Ridx,alias fun,Leafs...))
    {
        alias UNQs = staticMap!(getunq,Leafs);
        alias DMYs = staticMap!(getdmy,Leafs);
        alias getIndex = AliasSeq!(join([UNQs,DMYs]).onlyUniq(ignr),join([UNQs,DMYs]).onlyDummy(ignr));
    }
    else static if(is(Node == FnTensor!(idx,F),string idx,F))
    {
        alias getIndex = AliasSeq!(idx.onlyUniq(ignr),idx.onlyDummy(ignr));
    }
    else
    {
        alias getIndex = AliasSeq!("","");
    }
}

template getShapes(Node,string This,string ijk)
{
    static if(ijk.length == 1)
    {
        alias getShapes = Alias!(Node.getShape!(This,ijk));
    }
    else static if(ijk.length == 0)
    {
        alias getShapes = Alias!"";
    }
    else
    {
        alias getShapes = Alias!(getShapes!(Node,This,ijk[0..1]) ~ "," ~ getShapes!(Node,This,ijk[1..$]));
    }
}
template getExp(string This,Node,ulong N)
{
    static if(is(Node == Tree!(Ridx,L,R,OP,Leafs),string Ridx,L,R,string OP,Leafs...))
    {
            alias LHS = getExp!(This~"._lhs",L,N);
            alias RHS = getExp!(This~"._rhs",R,LHS[1]);
        static if(OP == "+" || OP =="-")
        {
            alias argsL  = Alias!(getIndex!(L,Ridx)[0].map!"[a]".join(",").to!string);
            alias argsR  = Alias!(getIndex!(R,Ridx)[0].map!"[a]".join(",").to!string);
            alias getExp = AliasSeq!(This~"._lhs.calc("~argsL~")"  ~ OP ~ This~"._rhs.calc("~argsR~")" ,RHS[1]);
        }
        else
        {
            alias getExp = AliasSeq!("("~LHS[0]~")" ~ OP ~ "("~RHS[0]~")",RHS[1]);
        }
    }
    else static if(is(Node == Leaf!(Ridx,idx,Ts),string Ridx,string idx,Ts))
    {
        alias ijk = Alias!(idx.map!(c=>[c]).join(",").to!string);
        alias getExp = AliasSeq!(This~".tensor["~ijk~"]",N+1);
    }
    else static if(is(Node == Func!(Ridx,fun,Leafs),string Ridx,alias fun,Leafs...))
    {
        alias getExp = AliasSeq!(Node.asExp!(This),N+1);
    }
    else static if(is(Node == FnTensor!(idx,F),string idx,F))
    {
        alias ijk = Alias!(idx.map!(c=>[c]).join(",").to!string);
        alias getExp = AliasSeq!(This~".FUN("~ijk~")",N+1);
    }
    else
    {
        alias getExp = AliasSeq!(This,N+1);
    }
}
template CommonTypeOfTensors(Leafs...)
{
    import std.traits;
    template getType(T)
    {
        static if(isBasicType!T)
            alias getType = T;
        else
            alias getType = T.Type;
    }
    alias Types = staticMap!(getType,filterTensors!Leafs);
    alias CommonTypeOfTensors = CommonType!(Types);
}

struct Tree(string ResultIdx ="",LHS,RHS,string op,Leafs...) 
{
    LHS _lhs;
    RHS _rhs;
    Leafs leafs;
    alias LeafTypes = Leafs;
    this(LHS lhs_,RHS rhs_,Leafs leafs_)
    {
        _lhs = lhs_;
        _rhs = rhs_;
        leafs = leafs_;
    }
    
    auto opBinary(string op,R)(R ohs) @nogc nothrow 
    {
        static if(is(R == Tree!(Ridx,Lhs,Rhs,OP,aLeafs),string Ridx,Lhs,Rhs,string OP,aLeafs...))
            return Tree!("",typeof(this),R,op,Leafs,aLeafs)(this,ohs,leafs,ohs.leafs);
        else
            return Tree!("",typeof(this),R,op,Leafs,R)(this,ohs,leafs,ohs);
    }
    
    auto opBinaryRight(string op,R)(R ohs) @nogc nothrow 
    {
        return Tree!("",R,typeof(this),op,R,Leafs)(ohs,this,ohs,leafs);
    }
    
    auto opUnary(string op)() @nogc nothrow 
    {
        return Tree!("",Type,typeof(this),op,Type,Leafs)(0,this,0,leafs);
    }

    alias Type = CommonTypeOfTensors!(Leafs);

    mixin(genCalc);
    static auto genCalc()
    {
        alias indexes  = getIndex!(typeof(this),ResultIdx);
        string UNIQ = (indexes[0].map!(c=>"size_t "~[c]).join(",").to!string);
        string DMMY = (indexes[1].map!(c=>[c]).join(",").to!string);
        string result= "auto calc("~ UNIQ ~") @nogc nothrow{\n";
        static if(false && (op=="+"||op=="-") && (getIndex!(LHS,ResultIdx)[1].length > 0 || getIndex!(RHS,ResultIdx)[1].length > 0))
        {
            string argsL  = getIndex!(LHS,ResultIdx)[0].map!"[a]".join(",").to!string;
            string argsR  = getIndex!(RHS,ResultIdx)[0].map!"[a]".join(",").to!string;
            result ~= "\t return _lhs.calc("~argsL~")" ~ op ~"_rhs.calc("~argsR~");\n";
            result ~="}\n";
        }
        else
        {
            result ~= "\t size_t["~indexes[1].length.to!string~"] shape = [" ~ getShapes!(typeof(this),"this",indexes[1]) ~"];\n";
            result ~= "\t auto result = cast(Type)0;\n";
            if(DMMY.length > 0)
            {
                static foreach(i,ijk;indexes[1])
                {
                    result ~= "\tforeach("~ijk~";0..shape["~i.to!string~"])\n";
                }
            }
            result~= "\t\tresult += "~getExp!("this",typeof(this),0)[0]~";\n";
            result~= "\treturn result;\n";
            result~= "}\n";
        }
       return result;
    }

    static auto getShape(string This = "this",string ijk)()
    {
        static foreach(i,Node;Leafs)
        {{ 
            static if(__traits(hasMember,Node,"getShape"))
            {
                string result = Node.getShape!(This~".leafs["~i.to!string~"]",ijk);
            }
            else
            {
                string result = "";
            }
            if(result != "")
            {
                return result;
            }
        }}
        return "";
    }

}



struct Leaf(string ResultIdx ="",string idx,aTensor)
{
    this(aTensor t)
    {
        tensor = t;
    }

    static auto getShape(string This = "this",string ijk)()
    {
        static if(idx.countUntil(ijk) >= 0)
        {
            return This ~ ".shape["~idx.countUntil(ijk).to!string ~"]";
        }
        else
        {
            return "";
        }
    }

    

    aTensor tensor;
    alias Type = aTensor.Type;
    @nogc nothrow typeof(this)[1] leafs() => [this];
    alias LeafTypes = AliasSeq!(typeof(this));


    @nogc nothrow auto shape() => tensor.shape;
    auto opBinary(string op,R)(R ohs)  @nogc nothrow 
    {
        static if(is(R == Tree!(Ridx,Lhs,Rhs,OP,Leafs),string Ridx,Lhs,Rhs,string OP,Leafs...))
            return Tree!("",typeof(this),R,op,typeof(this),Leafs)(this,ohs,this,ohs.leafs);
        else
            return Tree!("",typeof(this),R,op,typeof(this),R)(this,ohs,this,ohs);
    }
    auto opBinaryRight(string op,R)(R ohs)  @nogc nothrow 
    {
        static if(is(R == Tree!(Ridx,Lhs,Rhs,OP,Leafs),string Ridx,Lhs,Rhs,string OP,Leafs...))
            return Tree!("",R,typeof(this),op,Leafs,typeof(this))(this,ohs,ohs.leafs,this);
        else
            return Tree!("",R,typeof(this),op,R,typeof(this))(ohs,this,ohs,this);
    }
    auto opUnary(string op)() @nogc nothrow 
    {
        return Tree!("",Type,typeof(this),op,typeof(this))(cast(Type)0,this,this);
    }
    mixin(genCalc);
    static auto genCalc()
    {
        alias indexes  = getIndex!(typeof(this),ResultIdx);
        string UNIQ = (indexes[0].map!(c=>"size_t "~[c]).join(",").to!string);
        string DMMY = (indexes[1].map!(c=>[c]).join(",").to!string);
        string result="auto calc(" ~ UNIQ ~ ") @nogc nothrow {\n";
        result ~= "\t size_t["~indexes[1].length.to!string~"] shape = [" ~ getShapes!(typeof(this),"this",indexes[1]) ~"];\n";
        result ~= "\t auto result = cast(Type)0;\n";

        if(DMMY.length > 0)
        {
            static foreach(i,ijk;indexes[1])
            {
                result ~= "\tforeach("~ijk~";0..shape["~i.to!string~"])\n";
            }
        }
        alias ijk = Alias!(idx.map!(c=>[c]).join(",").to!string);
        result~= "\t\tresult += tensor["~ijk~"];\n";
        
        result~= "\treturn result;\n";
        result~= "}\n";
        return result;
    }

}


struct Func(string ResultIdx ="",alias fun,Leafs...)
{
    alias FUN = fun;
    static if(is(ReturnType!fun))
        alias Type = ReturnType!fun;
    else 
        alias Type = CommonTypeOfTensors!Leafs;
    alias ArgLength = Alias!(Leafs.length); 
    Leafs leafs;
    this(Leafs args_)
    {
        leafs = args_;
    }
    
    mixin(genCalc);
    static auto genCalc()
    {
        alias indexes  = getIndex!(typeof(this),ResultIdx);
        string UNIQ = (indexes[0].map!(c=>"size_t "~[c]).join(",").to!string);
        string DMMY = (indexes[1].map!(c=>[c]).join(",").to!string);
        string result="auto calc(" ~ UNIQ ~ ") @nogc nothrow {\n";
        result ~= "\t size_t["~indexes[1].length.to!string~"] shape = [" ~ getShapes!(typeof(this),"this",indexes[1]) ~"];\n";
        result ~= "\t auto result = cast(Type)0;\n";

        if(DMMY.length > 0)
        {
            static foreach(i,ijk;indexes[1])
            {
                result ~= "\tforeach("~ijk~";0..shape["~i.to!string~"])\n";
            }
        }
        result~= "\t\tresult += "~asExp~";\n";
        result~= "\treturn result;\n";
        result~= "}\n";
        return result;
    }
    static auto getShape(string This = "this",string ijk)()
    {
        static foreach(i,Node;Leafs)
        {{ 
            static if(__traits(hasMember,Node,"getShape"))
            {
                auto result = Node.getShape!(This~".leafs["~i.to!string~"]",ijk);
            }
            else
            {
                auto result = "";
            }
            if(result != "")
            {
                return result;
            }
        }}
        return "";
    }
    
    static auto asExp(string This = "this")()
    {
        string result = This~".FUN(";
        static foreach(i,arg;leafs)
        {{ 
            static if(is(Leafs[i] == Tree!(Ridx,LHS,RHS,op,Leafs_),string Ridx,LHS,RHS,string op,Leafs_...))
            {
                result ~= getExp!(This~".leafs["~i.to!string~"]",Leafs[i],0)[0] ~ ",";
            }
            else static if(is(Leafs[i] == Leaf!(Ridx,idx,Tns),string Ridx,string idx,Tns))
            {
                alias ijk = Alias!(idx.map!(c=>[c]).join(",").to!string);
                result ~= This~".leafs["~i.to!string~"].tensor[" ~ijk~"],";
            }
            else static if(is(Leafs[i] == Func!(Ridx,fun_,Leafs_),string Ridx,alias fun_,Leafs_...))
            {
                result ~= Leafs[i] .asExp!("leafs["~i.to!string~"]")~",";
            }
            else static if(is(Leafs[i] == FnTensor!(idx,F),string idx,F))
            {
                alias ijk = Alias!(idx.map!(c=>[c]).join(",").to!string);
                result ~= Leafs[i]~".leafs["~i.to!string~"].FUN("~ijk~"),";
            }
            else
            {
                result ~= This~".leafs["~i.to!string~"],";
            }
        }}
        result ~= ")";
        return result;
    }
    auto opBinary(string op,R)(R ohs)  @nogc nothrow 
    {
        static if(is(R == Tree!(Ridx,Lhs,Rhs,OP,Leafs_),string Ridx,Lhs,Rhs,string OP,Leafs_...))
            return Tree!("",typeof(this),R,op,typeof(this),Leafs_)(this,ohs,this,ohs.leafs);
        else
            return Tree!("",typeof(this),R,op,typeof(this),R)(this,ohs,this,ohs);
    }
    auto opBinaryRight(string op,R)(R ohs)  @nogc nothrow 
    {
        static if(is(R == Tree!(Ridx,Lhs,Rhs,OP,Leafs_),string Ridx,Lhs,Rhs,string OP,Leafs_...))
            return Tree!("",typeof(this),R,op,Leafs_,typeof(this))(this,ohs,ohs.leafs,this);
        else
            return Tree!("",typeof(this),R,op,R,typeof(this))(this,ohs,ohs,this);
    }
}

struct FnTensor(string idx,F)
{
    F FUN;
    this(F)(F fun) if (isCallable!F)
    {
        FUN = fun;
    }

    static auto getShape(string This = "this",string ijk)()
    {
        return "";
    }
    auto opBinary(string op,R)(R ohs) @nogc nothrow 
    {
        static if(is(R == Tree!(Ridx,Lhs,Rhs,OP,Leafs_),string Ridx,Lhs,Rhs,string OP,Leafs_...))
            return Tree!("",typeof(this),R,op,typeof(this),Leafs_)(this,ohs,this,ohs.leafs);
        else
            return Tree!("",typeof(this),R,op,typeof(this),R)(this,ohs,this,ohs);
    }
    auto opBinaryRight(string op,R)(R ohs) @nogc nothrow 
    {
        static if(is(R == Tree!(Ridx,Lhs,Rhs,OP,Leafs_),string Ridx,Lhs,Rhs,string OP,Leafs_...))
            return Tree!("",typeof(this),R,op,Leafs_,typeof(this))(this,ohs,ohs.leafs,this);
        else
            return Tree!("",typeof(this),R,op,R,typeof(this))(this,ohs,ohs,this);
    }
}

private auto calc(R)(R value)
{
    return value;
}