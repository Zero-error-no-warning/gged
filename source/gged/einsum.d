module ggeD.einsum;

import std;
import ggeD.ggeD;


class Einsum
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
        return new class{
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
        };
    }
}


// package(ggeD):
string onlyUniq(string input,string ignr="")
{
    return ignr != "" ? ignr : input.to!(dchar[]).filter!(a=>input.count(a) == 1).array.to!string;
}
string onlyDummy(string input,string ignr="")
{
    return ignr != "" ? input.to!(dchar[]).filter!(a=>ignr.count(a)==0).array.sort.uniq.array.to!string :  input.to!(dchar[]).filter!(a=>input.count(a) > 1 ).array.sort.uniq.array.to!string;
}

ulong[idx.length] getShape(string idx)(string[] idxes,ulong[][] shapes)
{
    assert(shapes.length == idxes.length,shapes.length.to!string ~" vs "~idxes.length.to!string);
    ulong[idx.length] result;
    foreach(k,c;idx)
    {
        foreach(i,ijk;idxes)
        {
            auto n = ijk.countUntil(c);
            if(n >= 0)
            {
                result[k] = shapes[i][n];
                break;
            }
        }
    }
    return result;
}
template getIdxList(Leafs...)
{
    static if(Leafs.length == 1)
    {
        static if(is(Leafs[0] == Leaf!(idx,T),string idx,T))
        {
            alias getIdxList = idx;
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
        static if(is(Leafs[0] == Leaf!(idx,T),string idx,T))
        {
            alias filterTensors = Leafs[0];
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
            static if(is(Node == Leaf!(idx,Ts),string idx,Ts))
                alias getIndex = AliasSeq!((LHS[0].onlyUniq(ignr)~RHS[0].onlyUniq(ignr)).array.sort.uniq.array.to!string ,"");
            else
                alias getIndex = RHS;
        }

    }
    else static if(is(Node == Leaf!(idx,Ts),string idx,Ts))
    {
        alias getIndex = AliasSeq!(idx.onlyUniq(ignr),idx.onlyDummy(ignr));
    }
    else
    {
        alias getIndex = AliasSeq!("","");
    }
}
template getExp(string This,Node,ulong N)
{
    static if(is(Node == Tree!(L,R,OP),L,R,string OP))
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
        alias getType = T.Type;
    }
    alias Types = staticMap!(getType,filterTensors!Leafs);
    alias CommonTypeOfTensors = CommonType!(Types);
}

class Tree(LHS,RHS,string op,Leafs...) 
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
    auto opBinary(string op,R)(R ohs)
    {
        static if(is(R == Tree!(Lhs,Rhs,OP,aLeafs),Lhs,Rhs,string OP,aLeafs...))
            return new Tree!(typeof(this),R,op,Leafs,aLeafs)(this,ohs,leafs,ohs.leafs);
        else
            return new Tree!(typeof(this),R,op,Leafs,R)(this,ohs,leafs,ohs);
    }
    auto opBinaryRight(string op,R)(R ohs)
    {
        return new Tree!(R,typeof(this),op,R,Leafs)(ohs,this,ohs,leafs);
    }
    auto opUnary(string op)()
    {
        return new Tree!(Type,typeof(this),op,Type,Leafs)(0,this,0,leafs);
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
                auto uniqshape = getShape!(indexes[0])([Idxes],This.shapeList);
                    auto result = gged!(Type)(uniqshape.tupleof);
            }
            else
            {
                Type result;
            }
            static if(indexes[1].length > 0)
            {
                auto dummyshape = getShape!(indexes[1])([Idxes],This.shapeList);
                auto sumgg = gged!(Type)(dummyshape.tupleof);
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

    static if((op == "+" || op == "-")&&!is(LHS==Type))
    {
        auto evalDummy(string ResultIdx)()
        {
            auto lhs =  _lhs.eval!ResultIdx;
            auto rhs =  _rhs.eval!ResultIdx;
            auto This = mixin("lhs"~op~"rhs");

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
                    auto uniqshape = getShape!(indexes[0])([Idxes],This.shapeList);
                    
                    auto result = gged!(Type)(uniqshape.tupleof);
                }
                else
                {
                    Type result;
                }
                static if(indexes[1].length > 0)
                {
                    auto dummyshape = getShape!(indexes[1])([Idxes],This.shapeList);
                    auto sumgg = gged!(Type)(dummyshape.tupleof);
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
    }
    else
    {
        auto evalDummy(string ResultIdx)()
        {
            auto lhs =  _lhs.evalDummy!ResultIdx;
            auto rhs =  _rhs.evalDummy!ResultIdx;
            auto This = mixin("lhs"~op~"rhs");
            return This;
        }
    }
    static auto once(Args...)(Args args)
    {
        bool result = true;
        static foreach(arg;args)
        {{
            result = result && (arg == 0);
        }}
        return result;
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

    auto shapeList()
    {
        ulong[][] shapes;
        static foreach(i,leaf;Leafs)
        {{
            static if(is(leaf == Leaf!(idx,T),string idx,T))
            {
                shapes ~= leafs[i].shape.dup;
            }
        }}
        return shapes;
    }
}
class Leaf(string idx,aTensor)
{
    this(aTensor t)
    {
        tensor = t;
    }

    aTensor tensor;
    alias Type = aTensor.Type;
    auto leafs() => [this];
    alias LeafTypes = AliasSeq!(typeof(this));

    auto shapeList()
    {
        return null;
    }

    auto shape() => tensor.shape;
    auto opBinary(string op,R)(R ohs)
    {
        static if(is(R == Tree!(Lhs,Rhs,OP,Leafs),Lhs,Rhs,string OP,Leafs...))
            return new Tree!(typeof(this),R,op,typeof(this),Leafs)(this,ohs,this,ohs.leafs);
        else
            return new Tree!(typeof(this),R,op,typeof(this),R)(this,ohs,this,ohs);
    }
    auto opBinaryRight(string op,R)(R ohs)
    {
        static if(is(R == Tree!(Lhs,Rhs,OP,Leafs),Lhs,Rhs,string OP,Leafs...))
            return new Tree!(R,typeof(this),op,Leafs,typeof(this))(this,ohs,ohs.leafs,this);
        else
            return new Tree!(R,typeof(this),op,R,typeof(this))(ohs,this,ohs,this);
    }
    auto opUnary(string op)()
    {
        return new Tree!(Type,typeof(this),op,typeof(this))(cast(Type)0,this,this);
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
            auto uniqshape = getShape!(uniq)([idx],[shape]);
            auto result = gged!(Type)(uniqshape.tupleof);
        }
        else
        {
            Type result;
        }
        static if(dmmy.length > 0)
        {
            auto dummyshape = getShape!(dmmy)([idx],[shape]);
            auto sumgg = gged!(Type)(dummyshape.tupleof);
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