module ggeD.einsum;

import std;
import ggeD.ggeD;

class br(alias fun)
{
    static:
    auto opCall(Args...)(Args args)
    {
        return new Func!(fun,Args)(args);
    }
}

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
        else static if(is(Leafs[0] == Func!(func,Leafs_),alias func,Leafs_...))
        {
            alias getIdxList = getIdxList!(Leafs_);
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
template filterTensorsValue(ulong N ,Leafs...)
{
    static if(Leafs.length == 1)
    {
        static if(is(Leafs[0] == Leaf!(idx,T),string idx,T) || isScalarType!(Leafs[0]))
        {
            alias filterTensorsValue = AliasSeq!(N+1,"_args["~N.to!string~"],");
        }
        else static if(is(Leafs[0] == Func!(func,Leafs_),alias func,Leafs_...))
        {
            alias r =  filterTensorsValue!(N,Leafs_);
            alias filterTensorsValue = AliasSeq!(r[0],"_args["~N.to!string~"]."~r[1]);
        }
        else 
        {
            alias filterTensorsValue  = AliasSeq!(N,"");
        }
    }
    else static if(Leafs.length == 0)
    {
        alias filterTensorsValue = AliasSeq!(N,"");
    }
    else
    {
        alias zero = filterTensorsValue!(N,Leafs[0]);
        alias rest = filterTensorsValue!(zero[0],Leafs[1..$]);
        alias filterTensorsValue = AliasSeq!(rest[0],zero[1] ~rest[1]);
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
    else static if(is(Node == Func!(fun,Leafs),alias fun,Leafs...))
    {
        alias text = Alias!(fun.stringof ~ "(");
        static foreach(i,leaf;Leafs)
        {
            alias exp = getExp!(This,leaf,N+i);
            text = Alias!(text ~ exp[0]~",");
        }
        text = Alias!(text ~")");
        alias getExp = AliasSeq!(text,N+Leafs.length);
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

            pragma(msg,genLoop!(ResultIdx,"This",typeof(This),indexes));
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
        auto lhs =  _lhs.evalDummy!ResultIdx;
        auto rhs =  _rhs.evalDummy!ResultIdx;
        auto This = mixin("lhs"~op~"rhs");
        return This;
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

class Func(alias fun,Leafs...)
{
    static if(is(ReturnType!fun))
        alias Type = ReturnType!fun;
    else 
        alias Type = CommonTypeOfTensors!Leafs;
    alias FunName = Alias!(""~fun.stringof);
    alias ArgLength = Alias!(Leafs.length); 
    Leafs _args;
    this(Leafs args_)
    {
        _args = args_;
    }
    auto evalDummy(string ignr="")()
    {
        static foreach(i,Leaf;Leafs)
        {
            mixin("auto arg_"~i.to!string ~" = _args["~i.to!string~"].evalDummy!ignr;");
        }
         
        mixin("return new Func!(fun,"~iota(Leafs.length).map!(i=>"typeof(arg_"~i.to!string~(i==Leafs.length-1 ? "": ",")).join~"))("~iota(Leafs.length).map!(i=>"arg_"~i.to!string~(i==Leafs.length-1 ? "":  ",")).join~");");
    }
    auto eval(string ignr="")()
    {
        auto This = evalDummy!ignr;
        alias Idxes = getIdxList!Leafs;
        alias uniq = Alias!(onlyUniq([Idxes].join, ignr));
        alias dmmy = Alias!(onlyDummy([Idxes].join, ignr));
        static if(uniq.length > 0)
        {
            writeln(uniq," , ",[Idxes]," , ",This.shapeList);
            auto uniqshape = getShape!(uniq)([Idxes],This.shapeList);
            auto result = gged!(Type)(uniqshape.tupleof);
        }
        else
        {
            Type result;
        }
        static if(dmmy.length > 0)
        {
            auto dummyshape = getShape!(dmmy)([Idxes],This.shapeList);
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
    
    static auto asExp(string This = "")()
    {
        string result = fun.stringof~"(";
        static foreach(i,arg;_args)
        {{ 
            static if(is(Leafs[i] == Tree!(LHS,RHS,op,Leafs_),LHS,RHS,string op,Leafs_...))
            {
                result ~= getExp!(This~"._args["~i.to!string~"]",Leafs[i],0) ~ ",";
            }
            else static if(is(Leafs[i] == Leaf!(idx,Tns),string idx,Tns))
            {
                alias ijk = Alias!(idx.replace("_","").map!(c=>[c]).join(",").to!string);
                result ~= This~"._args["~i.to!string~"].tensor[" ~ijk~"],";
            }
            else static if(is(Leafs[i] == Func!(fun_,Leafs_),alias fun_,Leafs_...))
            {
                result ~= Leafs[i] .asExp!("_args["~i.to!string~"]")~",";
            }
            else
            {
                result ~= This~"._args["~i.to!string~"],";
            }
        }}
        result ~= ")";
        return result;
    }
    auto shapeList()
    {
        ulong[][] shapes;
        static foreach(i,leaf;Leafs)
        {{
            static if(is(leaf == Leaf!(idx,T),string idx,T))
            {
                shapes ~= _args[i].shape.dup;
            }
            else static if(is(leaf == Func!(func,Leafs_),alias func,Leafs_...))
            {
                shapes ~= _args[i].shapeList;
            }
        }}
        return shapes;
    }
    template onlyLeafs()
    {
        alias onlyLeafs = AliasSeq!();
        static foreach(leaf;Leafs)
        {   
            static if(is(leaf == Leaf!(idx,Tns),string idx,Tns) || isBasicType!(leaf))
            {
                onlyLeafs = AliasSeq!(onlyLeafs,leaf);
            }
            else static if(is(leaf == Func!(func_,Leafs_),alias func_,Leafs_...))
            {
                onlyLeafs = AliasSeq!(onlyLeafs,leaf.onlyLeafs!());
            }
        }
    }
    auto opBinary(string op,R)(R ohs)
    {
        pragma(msg," return new Tree!(typeof(this),R,op,onlyLeafs!(),R)(this,ohs,"~filterTensorsValue!(0,Leafs)[1]~"ohs);");
        static if(is(R == Tree!(Lhs,Rhs,OP,Leafs_),Lhs,Rhs,string OP,Leafs_...))
            mixin("return new Tree!(typeof(this),R,op,onlyLeafs!(),Leafs_)(this,ohs,"~filterTensorsValue!(0,Leafs)[1]~"ohs.leafs);");
        else
            mixin(" return new Tree!(typeof(this),R,op,onlyLeafs!(),R)(this,ohs,"~filterTensorsValue!(0,Leafs)[1]~"ohs);");
    }
    auto opBinaryRight(string op,R)(R ohs)
    {
        static if(is(R == Tree!(Lhs,Rhs,OP,Leafs_),Lhs,Rhs,string OP,Leafs_...))
            return new Tree!(R,typeof(this),op,Leafs_,onlyLeafs!())(this,ohs,ohs.leafs,filterTensorsValue!_args);
        else
            return new Tree!(R,typeof(this),op,R,onlyLeafs!())(ohs,this,ohs,filterTensorsValue!_args);
    }
    //TODO: opBinary等を実装すること
}