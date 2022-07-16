module ggeD.einsum;
import ggeD;
import std;

package(ggeD)  string onlyUniq(string input)
{
    return input.to!(dchar[]).filter!(a=>input.count(a) == 1).array.to!string;
}

package(ggeD)  string onlyDup(string input)
{
    return input.to!(dchar[]).filter!(a=>input.count(a) > 1).array.sort.uniq.array.to!string;
}
package(ggeD)  string genNewOpBinary(ulong i)
{
    return "cast(rhs._mul[0].TYPE) fun("~iota(i).map!(x=>"a["~x.to!string~"],").join.to!string~")";
}

class Einsum
{
    static auto opBinary(string op, string Exp,X...)(lazy TensorIndexed!(Exp,X) rhs) if(op == "|")
    {
        auto result = rhs.eval;
        return Tensor!(TemplateArgsOf!(typeof(result))[1])(result._mul[0]);
    }
}

class BroadCast(alias f,Arg...)
{
    static auto opCall(string Exp,X...)(TensorIndexed!(Exp,X) tens, Arg arg)
    {
        auto newone = tens.eval();
        newone._mul[0] = newone._mul[0].dup;
        foreach(ref el;newone._mul[0].Elemental)
        {
            el = f(el,arg);
        }
        return newone;
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


package(ggeD)  struct TensorIndexed(string Exp,X...)
{
    
    package(ggeD)  X _mul;
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
        string unq = idx.join.onlyUniq;   // ijk
        string dup = idx.join.onlyDup;   // 
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
                        result ~= "_mul["~ r.front.to!string ~"].shape[" ~ r.map!(a=>idx[a].countUntil(c)).front.to!string ~"],";
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
                        result ~= "_mul["~ r.front.to!string ~"].shape[" ~ r.map!(a=>idx[a].countUntil(c)).front.to!string ~"],";
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
                        result~= ops[i] ~ " _mul[" ~ i.to!string ~ "][r_" ~ i.to!string ~ "]";
                    }
                    result ~=";\n";
                result ~= "}\n";
            result ~= "}\n";
            result ~= "return TensorIndexed!(`"~ unq ~"`,typeof(newgged))(newgged);\n";
        result ~= "}\n";
        return result;
    }

    private static string genEvalAdd(string exp1,string exp2,string op)
    {
        string[] idx = [exp1,exp2]; 
        string unq = idx.join.to!(dchar[]).sort.uniq.array.to!string;   // ijk
        string result;
            result ~="alias T = TemplateArgsOf!(X[0])[0];\n";
            result ~= "auto result = gged!T(";
                foreach(c;unq)
                {
                    auto r = iota(idx.length).filter!(a=>idx[a].countUntil(c) >= 0);
                    result ~= (r.front == 1 ? "rhs." : "lhs.") ~ "_mul[0].shape[" ~ r.map!(a=>idx[a].countUntil(c)).front.to!string ~"],";
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
                    result ~= "auto r1 = idx._idx;";
                    result ~= "auto r2 = idx._idx;";
                }
                result~="result[idx] = lhs._mul[0][r1] "~op~" rhs._mul[0][r2];";
            result ~= "}\n";
            result ~= "return TensorIndexed!(`"~ unq ~"`,typeof(result))(result);\n";
        return result;
    }

    
    auto opBinary(string op,string Exp2,Y...)(TensorIndexed!(Exp2,Y) rhs_) if(op == "+" || op == "-")
    {
        auto lhs = this.eval();
        auto rhs = rhs_.eval();
        static assert(TemplateArgsOf!(typeof(lhs))[0].to!(dchar[]).sort.array == TemplateArgsOf!(typeof(rhs))[0].to!(dchar[]).sort.array );
        mixin(genEvalAdd(Exp,Exp2,op));
        // pragma(msg,genEvalAdd(Exp,Exp2,op));
    }
    
    auto opBinary(string op,string Exp2,Y...)(TensorIndexed!(Exp2,Y) rhs) if(op == "*" || op == "/")
    {
        return TensorIndexed!(Exp ~op~ Exp2,AliasSeq!(X,Y))( _mul ,rhs._mul);
    }

    
    auto opUnary(string op)() if(op == "-")
    {
        auto lhs = this.eval();
        foreach(idx;lhs._mul[0])
        {
            lhs._mul[0][idx] = mixin(op ~"lhs._mul[0][idx]");
        }
        return lhs;
    }
    auto opBinary(string op,R)( R rhs)  if(isNumeric!R)
    {
        auto lhs = this.eval();
        foreach(idx;lhs._mul[0])
        {
            lhs._mul[0][idx] = mixin("lhs._mul[0][idx]" ~ op ~ "rhs");
        }
        return lhs;
    }
    auto opBinaryRight(string op, L)( L lhs)  if(isNumeric!L && op!="/")
    {
        auto rhs = this.eval();
        foreach(idx;rhs._mul[0])
        {
            rhs._mul[0][idx] = mixin("lhs" ~ op ~ "rhs._mul[0][idx]");
        }
        return rhs;
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