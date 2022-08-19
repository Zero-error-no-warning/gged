/*
Copyright (c) 2022 Zenw
Released under the MIT license
https://opensource.org/licenses/mit-license.php
*/

module ggeD.einsum;
import ggeD;
import std;

package(ggeD)  string onlyUniq(string input,string ignr)
{
    return ignr != "" ? ignr : input.to!(dchar[]).filter!(a=>input.count(a) == 1).array.array.to!string;
}

package(ggeD)  string onlyDup(string input,string ignr)
{
    return ignr != "" ? input.to!(dchar[]).filter!(a=>ignr.count(a)==0).array.sort.uniq.array.to!string :  input.to!(dchar[]).filter!(a=>input.count(a) > 1 ).array.sort.uniq.array.to!string;
}

class Einsum(Flag!"Parallel" para = No.Parallel)
{

    static auto opBinary(string op,T)(lazy T rhs) if(op == "|")
    {
        auto result = rhs.eval!para;
        static if(typeof(result).EXP.length == 0)
            return result._m[0][0];
        else
            return Tensor!(result.MainGG.TYPE,result.MainGG.RANK)(result._m[0]);
    }

    static auto opDispatch(string Ignr)()
    {
        return new class
        {
            static auto opBinary(string op,T)(lazy T rhs) if(op == "|")
            {
                alias type = TemplateOf!T;
                auto result = type!(Ignr~TemplateArgsOf!T[0],TemplateArgsOf!T[1..$])(rhs.tupleof).eval!para;
                static if(typeof(result).EXP.length == 0)
                    return result._m[0][0];
                else
                    return Tensor!(result.MainGG.TYPE,result.MainGG.RANK)(result._m[0]);
            }
        };
    }
}
auto broadCast(alias f,T)(T tensor)
{
    return BroadCast!("",f,T)(tensor);
}

template onlyOdd(X...)
{
    static if(X.length == 0) alias onlyEven = AliasSeq!();
    else static if(X.length <= 2) alias onlyOdd = AliasSeq!(X[0]);
    else
    {
        alias onlyOdd = AliasSeq!(X[0] , onlyOdd!(X[2..$]));
    }
}
template onlyEven(X...)
{
    static if(X.length <= 1) alias onlyEven = AliasSeq!();
    else static if(X.length == 2) alias onlyEven = AliasSeq!(X[1]);
    else
    {
        alias onlyEven = AliasSeq!(X[1] , onlyEven!(X[2..$]));
    }
}

struct BroadCast(string Ignr,alias f,T...) if(T.length %2 == 1)
{
    T[0] _m;
    onlyEven!(T[1..$]) _vs;
    alias _ops = onlyOdd!(T[1..$]);
    static string multireturn()
    {
        string result = "result";
        static foreach(i; 0.._vs.length)
        {{
            result ~= _ops[i] ~ "_vs["~i.to!string~"]";
        }}
        return result;
    }

    alias fun = unaryFun!f;
    auto eval(Flag!"Parallel" para = No.Parallel)()
    {
        alias type = TemplateOf!(T[0]);
        auto tmp = type!(Ignr,TemplateArgsOf!(T[0])[1..$])(_m.tupleof).eval!para;
        auto result = typeof(tmp)(tmp._m[0].dup);
        foreach(ref e ; result._m[0].Elemental)
        {
            e = fun(e);
        }
        return mixin(multireturn).eval!para;
    }

    auto opBinary(string op,X)(X rhs_) if(__traits(hasMember,X,"eval") && (op == "+" || op == "-"))
    {
        return TensorTree!(Ignr~TemplateArgsOf!X[0],op,typeof(this),typeof(rhs_))(this,rhs_);
    }
    auto opBinaryRight(string op,X)(X lhs_) if(__traits(hasMember,X,"eval") && (op == "+" || op == "-"))
    {
        return TensorTree!(TemplateArgsOf!X[0]~Ignr,op,typeof(lhs_),typeof(this))(lhs_,this);
    }

    auto opUnary(string op)() if(op == "-")
    {
        return BroadCast!(Ignr,x=>-fun(x),T)(_m);
    }
    auto opBinary(string op,R)( R rhs)  if( op=="*" || op=="/" )
    {
        return BroadCast!(Ignr,fun,T,op,R)(_m,rhs);
    }
    auto opBinaryRight(string op, L)( L lhs)  if(op=="*")
    {
        return BroadCast!(Ignr,fun,T,op,L)(_m,lhs);   
    }
}

package(ggeD) template myCommonType(X...)
{
    static if(X.length == 0) alias myCommonType = void;
    else static if(X.length == 1) alias myCommonType = X[0];
    else 
    {
        static if(is(typeof(X[0].init * X[1].init) U)) alias myCommonType = myCommonType!(U,X[2..$]);
        else alias myCommonType = void;
    }
}


package(ggeD)  struct TensorIndexed(string Ignr,string Exp,X...)
{
    package(ggeD) alias MainGG = X[0];
    package(ggeD) X _m;
    package(ggeD) alias TYPES =  X;
    package(ggeD) alias EXP = Alias!Exp;
    static if(Exp=="")
        private auto eval(Flag!"Parallel" para = No.Parallel)(){
            return this;
        }
    else{
        // pragma(msg,genEval);
        mixin(genEval);
    }
    private alias getTYPE(T) = T.TYPE;
    private static string genEval()
    {
        string[] idx = Exp.split!isOp;  // 
        string ops = "*" ~ Exp.filter!isOp.array.to!string;
        string unq = idx.join.onlyUniq(Ignr);   
        string dup = idx.join.onlyDup(Ignr);   
        string iall = idx.join.to!(dchar[]).array.sort.uniq.array.to!string;   

        alias T =  myCommonType!(staticMap!(getTYPE,X));
        string result;
        result ~= "private auto eval(Flag!`Parallel` para = No.Parallel)(){ // "~ Exp~"->"~ unq ~ "\n";
        if(Exp == unq)
        {
            result ~= "return this;\n}";
            return result;
        }
            result ~="alias T = myCommonType!(staticMap!(getTYPE,X));\n";

            result ~= "auto newgged = gged!T(";
                if(unq.length > 0)
                {
                    foreach(c;unq)
                    {
                        auto r = iota(idx.length).filter!(a=>idx[a].countUntil(c) >= 0);
                        result ~= "_m["~ r.front.to!string ~"].shape[" ~ r.map!(a=>idx[a].countUntil(c)).front.to!string ~"],";
                    }
                }
                else    // scalar
                {
                    result~="1";    
                }
            result ~= ");\n";

            result ~= "auto sumgg = gged!T(";
                    foreach(c;iall)
                    {
                        auto r = iota(idx.length).filter!(a=>idx[a].countUntil(c) >= 0);
                        result ~= "_m["~ r.front.to!string ~"].shape[" ~ r.map!(a=>idx[a].countUntil(c)).front.to!string ~"],";
                    }
            result ~= ");\n";

            result ~= "foreach(ui;newgged){\n";
                static if(isNumeric!T) result~="newgged[ui] = 0;\n";
                else static if(is(T==struct)) result~="newgged[ui] = T(0);\n";
                else static if(is(T==class))  result~="newgged[ui] = new T(0);\n";
            result ~= "}\n";
            foreach(i,c;iall)
            {
                auto r = iota(idx.length).filter!(a=>idx[a].countUntil(c) >= 0);
                result ~= "foreach(r_" ~ i.to!string~"; 0.."~"_m["~ r.front.to!string ~"].shape[" ~ r.map!(a=>idx[a].countUntil(c)).front.to!string ~"])\n";
            }
            result ~="{\n";
                result ~= "newgged[";
                if(unq.length>0)
                {
                    foreach(c;unq)
                    {
                        result~= "r_"~iall.countUntil(c).to!string~",";
                    }
                }
                else{
                    result ~= "0";
                } 
                result~="] += ";
                static if(isNumeric!T)          result ~= "1";
                else static if(is(T==struct))   result ~= "T(1)";
                else static if(is(T==class))    result ~= "new T(1)";
                foreach(i;0..idx.length)
                {
                    result ~=ops[i] ~ "_m["~i.to!string~"][";
                    foreach(c;idx[i])
                    {
                        result~= "r_"~iall.countUntil(c).to!string~",";
                    }
                    result ~="]";
                }
                result ~=";\n";
            result ~= "}\n";
            result ~= "return TensorIndexed!(`"~Ignr~"`,`"~ unq ~"`,typeof(newgged))(newgged);\n";
        result ~= "}\n";
        return result;
    }

    
    auto opBinary(string op,X)(X rhs_) if(__traits(hasMember,X,"eval") && (op == "+" || op == "-"))
    {
        return TensorTree!(Ignr~TemplateArgsOf!X[0],op,typeof(this),typeof(rhs_))(this,rhs_);
    }
    auto opBinaryRight(string op,X)(X _lhs) if(__traits(hasMember,X,"eval") && (op == "+" || op == "-"))
    {
        return TensorTree!(Ignr~TemplateArgsOf!X[0],op,typeof(_lhs),typeof(this))(_lhs,this);
    }

    auto opBinary(string op,string Exp2,string Ig2,Y...)(TensorIndexed!(Ig2,Exp2,Y) rhs) if(op == "*" || op == "/")
    {
        return TensorIndexed!(Ignr~Ig2,Exp ~op~ Exp2,AliasSeq!(X,Y))( _m ,rhs._m);
    }

    
    auto opUnary(string op)() if(op == "-")
    {
        auto result = typeof(this)(_m[0].dup);
        foreach(idx;_m[0])
        {
            result._m[0][idx] = mixin(op ~"_m[0][idx]");
        }
        return this;
    }
    auto opBinary(string op,R)( R rhs)  if(isNumeric!R && (op=="*" || op=="/"))
    {
        auto result = typeof(this)(_m[0].dup);
        foreach(idx;_m[0])
        {
            result._m[0][idx] = mixin("_m[0][idx]" ~ op ~ "rhs");
        }
        return result;
    }
    auto opBinaryRight(string op, L)( L lhs)  if(isNumeric!L && op=="*")
    {
        auto result = typeof(this)(_m[0].dup);
        foreach(idx;_m[0])
        {
            result._m[0][idx] = mixin("lhs" ~ op ~ "_m[0][idx]");
        }
        return result;
    }
}


package(ggeD) struct TensorTree(string Ignr="",string op,X...) if(X.length == 2)
{
    alias A = X[0];
    alias B = X[1];
    X _m;
    alias OP = Alias!op;
    alias IGNR = Alias!Ignr;
    alias typeA = A;
    alias typeB = B;

    auto opBinary(string op,X)(X rhs_) if(op == "+" || op == "-")
    {
        return TensorTree!(Ignr~TemplateArgsOf!X[0],op,typeof(this),typeof(rhs_))(this,rhs_);
    }
    auto opBinaryRight(string op,X)(X lhs_) if(op == "+" || op == "-")
    {
        return TensorTree!(Ignr~TemplateArgsOf!X[0],op,typeof(lhs_),typeof(this))(lhs_,this);
    }
    auto opUnary(string op)() if(op == "-")
    {
        return typeof(this)(-_m[0],-_m[1]);
    }
    auto opBinary(string op,R)( R rhs)  if(op=="*" ||op=="/")
    {
        return typeof(this)( mixin("_m[0]"~op~"rhs"),mixin("_m[1]"~op~"rhs"));
    }
    auto opBinaryRight(string op, L)( L lhs)  if( op=="*")
    {
        return typeof(this)( mixin("lhs"~ op~"_m[0]"),mixin("lhs"~op~"_m[1]"));
    }
    auto eval(Flag!"Parallel" para = No.Parallel)()
    {
        alias type1 = TemplateOf!A;
        alias type2 = TemplateOf!B;
        auto lhs = type1!(Ignr~TemplateArgsOf!A[0],TemplateArgsOf!A[1..$])(_m[0]._m).eval!para;
        auto rhs = type2!(Ignr~TemplateArgsOf!B[0],TemplateArgsOf!B[1..$])(_m[1]._m).eval!para;

        // static assert(TemplateArgsOf!(typeof(lhs))[0].to!(dchar[]).sort.array == TemplateArgsOf!(typeof(rhs))[0].to!(dchar[]).sort.array );
        static if(lhs.EXP.onlyUniq(Ignr).to!(dchar[]).sort == rhs.EXP.onlyUniq(Ignr).to!(dchar[]).sort)
        {
            // pragma(msg,genEvalAdd(lhs.EXP,rhs.EXP,op,Ignr));
            mixin(genEvalAdd(lhs.EXP,rhs.EXP,op,Ignr,para));
        }
        else
        {
            // return TensorIndexed!("","",Gged!bool(1)(true,1)); // dummy return;
            return lhs; // dummy return;
        }
    }
    private static string genEvalAdd(string exp1,string exp2,string op,string ignr,Flag!"Parallel" para = No.Parallel)
    {
        string[] idx = [exp1]; 
        // string unq = idx.join.to!(dchar[]).sort.uniq.array.to!string;   // ijk
        string unq = idx.join.onlyUniq(ignr);   
        string result;
        result ~= "//"~exp1~exp2~"\n";
            result ~= "alias T = TemplateArgsOf!(typeof(lhs).TYPES[0])[0];\n";
            result ~= "auto result = gged!T(";
            foreach(c;unq)
            {
                auto r = iota(idx.length).filter!(a=>idx[a].countUntil(c) >= 0);
                result ~="lhs._m[0].shape[" ~ r.map!(a=>idx[a].countUntil(c)).front.to!string ~"],";
            }
            result ~= ");\n";

            // foreach(i,c;unq)
            // {
            //     auto r = iota(idx.length).filter!(a=>idx[a].countUntil(c) >= 0);
            //     result ~= "foreach(r_" ~ i.to!string~"; 0.."~ (r.front == 1 ? "rhs." : "lhs.") ~"_m["~ r.front.to!string ~"].shape[" ~ r.map!(a=>idx[a].countUntil(c)).front.to!string ~"])\n";
            // }
            result ~= "foreach(";
            foreach(i,c;unq)
            {
                result ~= "r_" ~ i.to!string~",";
            }
            result ~="; result" ~ (para ? "" : ".Serial") ~ ")\n";
            result ~= "{\n";
            result ~= "result[";
            foreach(c;unq)
            {
                result~= "r_"~unq.countUntil(c).to!string~",";
            }
            result~="]=";
            result~="lhs._m[0][";
            foreach(c;unq)
            {
                result~= "r_"~exp1.countUntil(c).to!string~",";
            }
            result ~="]"~op~" rhs._m[0][";
            foreach(c;unq)
            {
                result~= "r_"~exp2.countUntil(c).to!string~",";
            }
            result ~="];\n";
            result ~= "}\n";
            result ~= "return TensorIndexed!(`"~Ignr~"`,`"~ unq ~"`,typeof(result))(result);\n";
        return result;
    }

    
}
package(ggeD)  bool isOp(dchar op)
{
    return op == '*' || op =='/';
}
package(ggeD)  bool isOpPlusMinus(string op)
{
    return op == "+" || op =="-";
}