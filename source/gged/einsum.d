module ggeD.einsum;

import std;
import ggeD.ggeD;


unittest
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


auto br(alias fun,Args...)(Args args)
{
    return Func!(fun,Args)(args);
}


auto fnTensor(F)(F fun) if(isCallable!F)
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
    
    auto opBinary(string op,R)(R rhs) if(op == "|")
    {
        static if(getIndex!(R)[0].length > 0)
        {
            return rhs.eval.tensor;
        }
        else
        {
            return rhs.eval;
        }
    }
    
    auto opDispatch(string name)()
    { 
        struct Sub
        {
            auto opBinary(string op,R)(R rhs) if(op == "|")
            {
                static if(getIndex!(R,name)[0].length > 0)
                {
                    return rhs.eval!name.tensor;
                }
                else
                {
                    return rhs.eval!name;
                }
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


// ulong[idx.length] getShape(string idx)(string[] idxes,ulong[][] shapes)
// {
//     ulong[idx.length] result;
//     foreach(k,c;idx)
//     {
//         foreach(i,ijk;idxes)
//         {
//             auto n = ijk.countUntil(c);
//             if(n >= 0)
//             {
//                 result[k] = shapes[i][n];
//                 break;
//             }
//         }
//     }
//     return result;
// }

template getIdxList(Leafs...)
{
    static if(Leafs.length == 1)
    {
        static if(is(Leafs[0] == Leaf!(idx,T),string idx,T))
        {
            alias getIdxList = idx;
        }
        else static if(is(Leafs[0] == Func!(func,Leafs_),alias func,Leafs_...))
        {
            alias getIdxList = getIdxList!(Leafs_);
        }
        else static if(is(Leafs[0] == FnTensor!(idx,F),string idx,F))
        {
            alias getIdxList = idx;
        }
        else static if(is(Leafs[0] == Tree!(LHS,RHS,op,Leafs_) ,LHS,RHS,string op,Leafs_...))
        {
            alias getIdxList = getIdxList!Leafs_;
        }
        else 
        {
            alias getIdxList = AliasSeq!();
        }
    }
    else static if(Leafs.length == 0)
    {
        alias getIdxList = AliasSeq!();
    }
    else
    {
        alias getIdxList = AliasSeq!(getIdxList!(Leafs[0]),getIdxList!(Leafs[1..$]));
    }
}

template filterTensors(Leafs...)
{
    static if(Leafs.length == 1)
    {
        static if(is(Leafs[0] == Leaf!(idx,T),string idx,T) || isBasicType!(Leafs[0]))
        {
            alias filterTensors = Leafs[0];
        }
        else static if(is(Leafs[0] == Func!(func,Leafs_),alias func,Leafs_...))
        {
            alias filterTensors = filterTensors!(Leafs_);
        }
        else static if(is(Leafs[0] == Tree!(L,R,OP,Leafs_),L,R,string OP,Leafs_...))
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
    static if(is(Node == Tree!(L,R,OP,Leafs),L,R,string OP,Leafs...))
    {
        alias LHS = getIndex!(L,ignr);
        alias RHS = getIndex!(R,ignr);
        static if(OP == "*" || OP == "/")
        {
            alias getIndex = AliasSeq!((LHS[0]~LHS[1]~RHS[0]~RHS[1]).onlyUniq(ignr),(LHS[0]~LHS[1]~RHS[0]~RHS[1]).onlyDummy(ignr));
        }
        else 
        {
            alias getIndex = AliasSeq!((LHS[0].onlyUniq(ignr)~RHS[0].onlyUniq(ignr)).array.sort.uniq.array.to!string ,"");
        }
    }
    else static if(is(Node == Leaf!(idx,Ts),string idx,Ts))
    {
        alias getIndex = AliasSeq!(idx.onlyUniq(ignr),idx.onlyDummy(ignr));
    }
    else static if(is(Node == Func!(fun,Leafs),alias fun,Leafs...))
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
    else
    {
        alias getShapes = Alias!(getShapes!(Node,This,ijk[0..1]) ~ "," ~ getShapes!(Node,This,ijk[1..$]));
    }
}
template getExp(string This,Node,ulong N)
{
    static if(is(Node == Tree!(L,R,OP,Leafs),L,R,string OP,Leafs...))
    {
        alias LHS = getExp!(This,L,N);
        alias RHS = getExp!(This,R,LHS[1]);
        alias getExp = AliasSeq!("("~LHS[0]~")" ~ OP ~ "("~RHS[0]~")",RHS[1]);
    }
    else static if(is(Node == Leaf!(idx,Ts),string idx,Ts))
    {
        alias ijk = Alias!(idx.replace("_","").map!(c=>[c]).join(",").to!string);
        alias getExp = AliasSeq!(This~".leafs["~N.to!string~"].tensor["~ijk~"]",N+1);
    }
    else static if(is(Node == Func!(fun,Leafs),alias fun,Leafs...))
    {
        alias getExp = AliasSeq!(Node.asExp!(This~".leafs["~N.to!string~"]"),N+1);
    }
    else static if(is(Node == FnTensor!(idx,F),string idx,F))
    {
        alias ijk = Alias!(idx.replace("_","").map!(c=>[c]).join(",").to!string);
        alias getExp = AliasSeq!(This~".leafs["~N.to!string~"].FUN("~ijk~")",N+1);
    }
    else
    {
        alias getExp = AliasSeq!(This~".leafs["~N.to!string~"]",N+1);
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

struct Tree(LHS,RHS,string op,Leafs...) 
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
    
    @nogc auto opBinary(string op,R)(R ohs)
    {
        static if(is(R == Tree!(Lhs,Rhs,OP,aLeafs),Lhs,Rhs,string OP,aLeafs...))
            return Tree!(typeof(this),R,op,Leafs,aLeafs)(this,ohs,leafs,ohs.leafs);
        else
            return Tree!(typeof(this),R,op,Leafs,R)(this,ohs,leafs,ohs);
    }
    
    @nogc auto opBinaryRight(string op,R)(R ohs)
    {
        return Tree!(R,typeof(this),op,R,Leafs)(ohs,this,ohs,leafs);
    }
    
    @nogc auto opUnary(string op)()
    {
        return Tree!(Type,typeof(this),op,Type,Leafs)(0,this,0,leafs);
    }

    alias Type = CommonTypeOfTensors!(Leafs);
    
    auto eval(string ResultIdx="")()
    {
        auto This = this.evalDummy!ResultIdx();
        static if(is(typeof(This) : Type) || is(typeof(This) == Leaf!(idx,Tns),string idx, Tns))
        {
            return This;
        }
        else
        {
            alias Idxes = getIdxList!(This.LeafTypes);
            alias indexes  = getIndex!(typeof(This),ResultIdx);
            static if(indexes[0].length > 0)
            {
                auto result = mixin("gged!(Type)("~getShapes!(typeof(This),"This",indexes[0])~")");
            }
            else
            {
                Type result;
            }
            static if(indexes[1].length > 0)
            {
                auto sumgg = mixin("gged!(Empty)("~getShapes!(typeof(This),"This",indexes[1])~")");
            }
            mixin(genLoop!(ResultIdx,"This",typeof(This),indexes));

            static if(indexes[0].length > 0)
            {
                auto leaf = result.opDispatch!(indexes[0]);
            }
            else
            {
                auto leaf = result;
            }
            return leaf;
        }

    }

    
    auto evalDummy(string ResultIdx)()
    {
        alias idxL  = getIndex!((LHS),ResultIdx);
        alias idxR  = getIndex!((RHS),ResultIdx);
        static if (idxL[0] == idxR[0])
        {
            auto lhs =  _lhs.eval!ResultIdx;
            auto rhs =  _rhs.eval!ResultIdx;
        }
        else
        {
            auto lhs =  _lhs.evalDummy!ResultIdx;
            auto rhs =  _rhs.evalDummy!ResultIdx;
        }
        auto This = mixin("lhs"~op~"rhs");
        return This;
    }
    static auto genLoop(string ResultIdx="",string This,TypeThis,indexes...)()
    {
        string UNIQ = (indexes[0].map!(c=>[c]).join(",").to!string);
        string DMMY = (indexes[1].map!(c=>[c]).join(",").to!string);
        string result="";
        if(UNIQ.length > 0)
        {
            result~= "foreach("~UNIQ~";"~"result.index){\n";
            result~= "\tresult["~UNIQ~"] = 0.;\n";
        }
        else
        {
            result~= "result = 0.;\n";
        }
        if(DMMY.length > 0)
        {
            result~= "\tforeach("~DMMY~";"~"sumgg.index){\n";
        }
        if(UNIQ.length > 0)
        {
            result~= "\t\tresult["~UNIQ~"] += "~getExp!(This,TypeThis,0)[0]~";\n";
        }
        else
        {
            result~= "\t\tresult += "~getExp!(This,TypeThis,0)[0]~";\n";
        }
        if(DMMY.length > 0)
        {
            result~= "\t}\n";
        }
        if(UNIQ.length > 0)
        {
            result~="}";
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



struct Leaf(string idx,aTensor)
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
    auto leafs() => [this];
    alias LeafTypes = AliasSeq!(typeof(this));


    auto shape() => tensor.shape;
    @nogc auto opBinary(string op,R)(R ohs)
    {
        static if(is(R == Tree!(Lhs,Rhs,OP,Leafs),Lhs,Rhs,string OP,Leafs...))
            return Tree!(typeof(this),R,op,typeof(this),Leafs)(this,ohs,this,ohs.leafs);
        else
            return Tree!(typeof(this),R,op,typeof(this),R)(this,ohs,this,ohs);
    }
    @nogc auto opBinaryRight(string op,R)(R ohs)
    {
        static if(is(R == Tree!(Lhs,Rhs,OP,Leafs),Lhs,Rhs,string OP,Leafs...))
            return Tree!(R,typeof(this),op,Leafs,typeof(this))(this,ohs,ohs.leafs,this);
        else
            return Tree!(R,typeof(this),op,R,typeof(this))(ohs,this,ohs,this);
    }
    @nogc auto opUnary(string op)()
    {
        return Tree!(Type,typeof(this),op,typeof(this))(cast(Type)0,this,this);
    }
    auto evalDummy(string ignr="")()
    {
        return this;
    }
    auto eval(string ignr="")()
    { 
        alias uniq = Alias!(idx.onlyUniq(ignr));
        alias dmmy = Alias!(idx.onlyDummy(ignr));
        static if(uniq.length > 0)
        {
            auto result = mixin("gged!(Type)("~getShapes!(typeof(this),"this",uniq)~")");
        }
        else
        {
            Type result;
        }
        static if(dmmy.length > 0)
        {
            auto sumgg = mixin("gged!(Empty)("~getShapes!(typeof(this),"this",dmmy)~")");
        }


        mixin(genLoop!(ignr,uniq,dmmy));

        static if(uniq.length > 0)
        {
            auto leaf = result.opDispatch!(uniq);
        }
        else
        {
            auto leaf = result;
        }
        return leaf;
    }
    static auto genLoop(string ResultIdx="",indexes...)()
    {
        string UNIQ = (indexes[0].map!(c=>[c]).join(",").to!string);
        string DMMY = (indexes[1].map!(c=>[c]).join(",").to!string);
        string result="";
        if(UNIQ.length > 0)
        {
            result~= "foreach("~UNIQ~";"~"result.index){\n";
            result~= "\tresult["~UNIQ~"] = 0.;\n";
        }
        else
        {
            result~= "result = 0.;\n";
        }
        if(DMMY.length > 0)
        {
            result~= "\tforeach("~DMMY~";"~"sumgg.index){\n";
        }
        if(UNIQ.length > 0)
        {
            result~= "\t\tresult["~UNIQ~"] += "~getExp!("this",typeof(this),0)[0]~";\n";
        }
        else
        {
            result~= "\t\tresult += "~getExp!("this",typeof(this),0)[0]~";\n";
        }
        if(DMMY.length > 0)
        {
            result~= "\t}\n";
        }
        if(UNIQ.length > 0)
        {
            result~="}";
        }
       return result;
    }
}

auto evalDummy(string ignr="",T)(T value)
{
    return value;
}

auto eval(string ignr="",T)(T value)
{
    return value;
}


struct Func(alias fun,Leafs...)
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
    auto evalDummy(string ignr="")()
    {
        static foreach(i,Leaf;Leafs)
        {
            mixin("auto arg_"~i.to!string ~" = leafs["~i.to!string~"].evalDummy!ignr;");
        }
        mixin("return Func!(fun,"~iota(Leafs.length).map!(i=>"typeof(arg_"~i.to!string~"),").join~")("~iota(Leafs.length).map!(i=>"arg_"~i.to!string~",").join~");");
    }
    auto eval(string ignr="")()
    {
        auto This = evalDummy!ignr;
        alias Idxes = getIdxList!Leafs;
        alias uniq = Alias!(onlyUniq([Idxes].join, ignr));
        alias dmmy = Alias!(onlyDummy([Idxes].join, ignr));
        static if(uniq.length > 0)
        {
            auto result = mixin("gged!(Type)("~getShapes!(typeof(This),"this",uniq)~")");
        }
        else
        {
            Type result;
        }
        static if(dmmy.length > 0)
        {
            auto sumgg = mixin("gged!(Empty)("~getShapes!(typeof(This),"this",dmmy)~")");
        }
        mixin(genLoop!(ignr,uniq,dmmy));

        static if(uniq.length > 0)
        {
            auto leaf = result.opDispatch!(uniq);
        }
        else
        {
            auto leaf = result;
        }
        return leaf;
    }
    static auto genLoop(string ResultIdx="",indexes...)()
    { 
        string UNIQ = (indexes[0].map!(c=>[c]).join(",").to!string);
        string DMMY = (indexes[1].map!(c=>[c]).join(",").to!string);
        string result="";
        if(UNIQ.length > 0)
        {
            result~= "foreach("~UNIQ~";"~"result.index){\n";
            result~= "\tresult["~UNIQ~"] = 0.;\n";
        }
        else
        {
            result~= "result = 0.;\n";
        }
        if(DMMY.length > 0)
        {
            result~= "\tforeach("~DMMY~";"~"sumgg.index){\n";
        }
        if(UNIQ.length > 0)
        {
            result~= "\t\tresult["~UNIQ~"] += ";
        }
        else
        {
            result~= "\t\tresult+=";
        }
        result ~= asExp ~ ";\n";
        if(DMMY.length > 0)
        {
            result~= "\t}\n";
        }
        if(UNIQ.length > 0)
        {
            result~="}";
        }
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
            static if(is(Leafs[i] == Tree!(LHS,RHS,op,Leafs_),LHS,RHS,string op,Leafs_...))
            {
                result ~= getExp!(This~".leafs["~i.to!string~"]",Leafs[i],0)[0] ~ ",";
            }
            else static if(is(Leafs[i] == Leaf!(idx,Tns),string idx,Tns))
            {
                alias ijk = Alias!(idx.replace("_","").map!(c=>[c]).join(",").to!string);
                result ~= This~".leafs["~i.to!string~"].tensor[" ~ijk~"],";
            }
            else static if(is(Leafs[i] == Func!(fun_,Leafs_),alias fun_,Leafs_...))
            {
                result ~= Leafs[i] .asExp!("leafs["~i.to!string~"]")~",";
            }
            else static if(is(Leafs[i] == FnTensor!(idx,F),string idx,F))
            {
                alias ijk = Alias!(idx.replace("_","").map!(c=>[c]).join(",").to!string);
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
    @nogc auto opBinary(string op,R)(R ohs)
    {
        static if(is(R == Tree!(Lhs,Rhs,OP,Leafs_),Lhs,Rhs,string OP,Leafs_...))
            return Tree!(typeof(this),R,op,typeof(this),Leafs_)(this,ohs,this,ohs.leafs);
        else
            return Tree!(typeof(this),R,op,typeof(this),R)(this,ohs,this,ohs);
    }
    @nogc auto opBinaryRight(string op,R)(R ohs)
    {
        static if(is(R == Tree!(Lhs,Rhs,OP,Leafs_),Lhs,Rhs,string OP,Leafs_...))
            return Tree!(typeof(this),R,op,Leafs_,typeof(this))(this,ohs,ohs.leafs,this);
        else
            return Tree!(typeof(this),R,op,R,typeof(this))(this,ohs,ohs,this);
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
    @nogc auto opBinary(string op,R)(R ohs)
    {
        static if(is(R == Tree!(Lhs,Rhs,OP,Leafs_),Lhs,Rhs,string OP,Leafs_...))
            return Tree!(typeof(this),R,op,typeof(this),Leafs_)(this,ohs,this,ohs.leafs);
        else
            return Tree!(typeof(this),R,op,typeof(this),R)(this,ohs,this,ohs);
    }
    @nogc auto opBinaryRight(string op,R)(R ohs)
    {
        static if(is(R == Tree!(Lhs,Rhs,OP,Leafs_),Lhs,Rhs,string OP,Leafs_...))
            return Tree!(typeof(this),R,op,Leafs_,typeof(this))(this,ohs,ohs.leafs,this);
        else
            return Tree!(typeof(this),R,op,R,typeof(this))(this,ohs,ohs,this);
    }
}

struct Empty
{

}