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
    return ignr != "" ? ignr : input.to!(dchar[]).filter!(a=>input.count(a) == 1).array.sort.uniq.array.to!string;
}

package(ggeD)  string onlyDup(string input,string ignr)
{
    return input.to!(dchar[]).filter!(a=>input.count(a) > 1 && ignr.to!(dchar[]).all!(b=>a!=b)).array.sort.uniq.array.to!string;
}

class Einsum
{
    static auto opBinary(string op,T)(lazy T rhs) if(op == "|")
    {
        auto result = rhs.eval;
        static if(typeof(result).EXP.length == 0)
            return result._m[0][0];
        else
            return Tensor!(result.MainGG)(result._m[0]);
    }

    static auto opDispatch(string Ignr)()
    {
        return new class
        {
            static auto opBinary(string op,T)(lazy T rhs) if(op == "|")
            {
                alias type = TemplateOf!T;
                auto result = type!(Ignr~TemplateArgsOf!T[0],TemplateArgsOf!T[1..$])(rhs._m).eval;
                static if(typeof(result).EXP.length == 0)
                    return result._m[0][0];
                else
                    return Tensor!(result.MainGG)(result._m[0]);
            }
        };
    }
}
auto broadCast(alias f,T)(T tensor)
{
    return BroadCast!("",f,T)(tensor);
}
struct BroadCast(string Ignr,alias f,T)
{
    T _m;
    alias fun = unaryFun!f;
    auto eval()
    {
        alias type = TemplateOf!T;
        auto result = type!(Ignr,TemplateArgsOf!T[1..$])(_m.tupleof).eval;
        foreach(ref e;result._m[0].Elemental)
        {
            e = fun(e);
        }
        return result;
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
        return typeof(this)(-_m);
    }
    auto opBinary(string op,R)( R rhs)  if(op=="*" ||op=="/" )
    {
        return typeof(this)( mixin("_m"~op~"rhs"));
    }
    auto opBinaryRight(string op, L)( L lhs)  if(op=="*")
    {
        return typeof(this)( mixin("lhs"~ op~"_m"));
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
        private auto eval(){
            return this;
        }
    else{
        mixin(genEval);
    }
    private alias getTYPE(T) = T.TYPE;
    private static string genEval()
    {
        string[] idx = Exp.split!isOp;  // 
        string ops = "*" ~ Exp.filter!isOp.array.to!string;
        string unq = idx.join.onlyUniq(Ignr);   
        string dup = idx.join.onlyDup(Ignr);   

        alias T =  myCommonType!(staticMap!(getTYPE,X));
        string result;
        result ~= "private auto eval(){ // "~ Exp~"->"~ unq ~ "\n";
        if(Exp == unq)
        {
            result ~= "return this;\n}";
            return result;
        }
            // result ~="alias T = TemplateArgsOf!(X[0])[0];\n";
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
                if(dup.length > 0)
                {
                    foreach(c;dup)
                    {
                        auto r = iota(idx.length).filter!(a=>idx[a].countUntil(c) >= 0);
                        result ~= "_m["~ r.front.to!string ~"].shape[" ~ r.map!(a=>idx[a].countUntil(c)).front.to!string ~"],";
                    }
                }
                else    // scalar
                {
                    result~="1";    // コメント
                }
            result ~= ");\n";

            result ~= "foreach(ui;newgged){\n";
                static if(isNumeric!T) result~="newgged[ui] = 0;\n";
                else static if(is(T==struct)) result~="newgged[ui] = T(0);\n";
                else static if(is(T==class))  result~="newgged[ui] = new T(0);\n";
                result~="foreach(di;sumgg.Serial){\n";
                    foreach(i,ijk;idx)
                    {
                        if(ijk.length > 1)
                        {
                            result ~= "auto r_" ~ i.to!string ~ " = Vec!("~ ijk.length.to!string ~",ulong)([";
                                foreach(c;ijk)
                                {
                                    auto cnt =  unq.countUntil(c);
                                    auto len = cnt >= 0 ? unq.length : dup.length;
                                    if(len>1) result ~= (cnt>=0 ? "ui["~cnt.to!string~"]._idx" : "di["~dup.countUntil(c).to!string~"]._idx") ~",";
                                    else result ~= (cnt>=0 ? "ui._idx" : "di._idx") ~",";
                                }
                            result ~= "]);\n";
                        }
                        else
                        {
                            result ~= "auto r_" ~ i.to!string ~ " = ";
                            auto cnt =  unq.countUntil(ijk[0]);
                            auto len = cnt >= 0 ? unq.length : dup.length;
                            if(len>1) result ~= (cnt>=0 ? "ui["~cnt.to!string~"]._idx" : "di["~dup.countUntil(ijk[0]).to!string~"]._idx") ~";\n";
                            else result ~= (cnt>=0 ? "ui._idx" : "di._idx") ~";\n";
                        }
                    }
                    static if(isNumeric!T)  result~="newgged[ui] =  newgged[ui] + 1";
                    else static if(is(T==struct)) result~="newgged[ui]  =newgged[ui] + T(1)";
                    else static if(is(T==class)) result~=" newgged[ui] = newgged[ui] + new T(1)";
                    foreach(i;0..idx.length)
                    {
                        result~= ops[i] ~ " _m[" ~ i.to!string ~ "][r_" ~ i.to!string ~ "]";
                    }
                    result ~=";\n";
                result ~= "}\n";
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
        return this;
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
    auto eval()
    {
        alias type1 = TemplateOf!A;
        alias type2 = TemplateOf!B;
        auto lhs = type1!(Ignr~TemplateArgsOf!A[0],TemplateArgsOf!A[1..$])(_m[0]._m).eval;
        auto rhs = type2!(Ignr~TemplateArgsOf!B[0],TemplateArgsOf!B[1..$])(_m[1]._m).eval;

        // static assert(TemplateArgsOf!(typeof(lhs))[0].to!(dchar[]).sort.array == TemplateArgsOf!(typeof(rhs))[0].to!(dchar[]).sort.array );
        static if(lhs.EXP.onlyUniq(Ignr).to!(dchar[]).sort == rhs.EXP.onlyUniq(Ignr).to!(dchar[]).sort)
        {
            mixin(genEvalAdd(lhs.EXP,rhs.EXP,op,Ignr));
        }
        else
        {
            // return TensorIndexed!("","",Gged!bool(1)(true,1)); // dummy return;
            return lhs; // dummy return;
        }
    }
    private static string genEvalAdd(string exp1,string exp2,string op,string ignr)
    {
        string[] idx = [exp1]; 
        // string unq = idx.join.to!(dchar[]).sort.uniq.array.to!string;   // ijk
        string unq = idx.join.onlyUniq(ignr);   
        string result;
            result ~= "alias T = TemplateArgsOf!(typeof(lhs).TYPES[0])[0];\n";
            result ~= "auto result = gged!T(";
            if(unq.length > 0)
            {
                foreach(c;unq)
                {
                    auto r = iota(idx.length).filter!(a=>idx[a].countUntil(c) >= 0);
                    result ~= (r.front == 1 ? "rhs." : "lhs.") ~ "_m[0].shape[" ~ r.map!(a=>idx[a].countUntil(c)).front.to!string ~"],";
                }
            }
            else    // scalar
            {
                result~="1";    
            }
            result ~= ");\n";

            result ~= "foreach(idx;result){\n";
                if(unq.length > 1)
                {
                    result ~= "auto r1 = Vec!("~ exp1.length.to!string ~",ulong)([";
                        foreach(c;unq)
                        {

                            auto cnt =  exp1.countUntil(c);
                            if(unq.length > 1) result ~= "idx["~cnt.to!string~"]._idx,";
                            else result ~=  "idx._idx,";
                        }
                    result ~= "]);\n";
                    result ~= "auto r2 = Vec!("~ exp2.length.to!string ~",ulong)([";
                        foreach(c;unq)
                        {
                            auto cnt =  exp2.countUntil(c);
                            result ~= "idx["~cnt.to!string~"]._idx,";
                        }
                    result ~= "]);\n";
                }
                else
                {
                    result ~= "ulong r1 = idx;\n";
                    result ~= "ulong r2 = idx;\n";
                }
                result~="result[idx] = lhs._m[0][r1] "~op~" rhs._m[0][r2];\n";
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