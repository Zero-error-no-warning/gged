module ggeD.vec;
import std;
import ggeD;

struct Vec(ulong Dim,T = double,Names...) if (Names.length <= Dim)
{
    T[Dim] _value;
    alias _value this;
    alias names = Names;

    static if(Names.length > 0)  
    {
        static string makeElemName()
        {
            string result = "static enum _elem = [";
            static foreach(i,name;Names)
            {
                result ~=  `"` ~ name ~`" : ` ~ i.to!string ~ " , ";
            }
            result ~= "];\n";
            return result;
        }
        mixin(makeElemName());

        ref T opDispatch(string member)()
        {
            return _value[_elem[member]];
        }
        ref auto opIndex(string name)
        {
            return _value[_elem[name]];
        }
    }

    this(X)(const X[Dim] value_)
    {
        _value = value_;
    }

    this(X)(X[] value_) in
    {
        assert(value_.length == Dim);
    }do
    {
        static foreach(i;0..Dim)
        {
            _value[i] = value_[i];
        }
    }

    this(X...)(X value_) if(X.length == Dim)
    {
        static foreach(i;0..Dim)
        {
            _value[i] = value_[i];
        }
    }
    this(Gged!(T,1) gg)
    {
        static foreach(i;0..Dim)
        {
            _value[i] = gg.elements[i];
        }
    }

    this(X)(X value_) if(!isArray!X)
    {
        static foreach(i;0..Dim)
        {
            _value[i] = value_;
        }
    }

    auto opBinary(string op,R,N...)(const Vec!(Dim,R,N) rhs) const 
    {
        alias S = CommonType!(T,R);
        static if(op == "%"){
            return iota(Dim).map!(idx=>_value[idx]*rhs._value[idx]).reduce!"a+b";
        }
        static if(op == "+"){
            return Vec!(Dim,S,Names)(iota(Dim).map!(idx=>_value[idx] + rhs._value[idx]).array);
        }
        static if(op == "-"){
            return Vec!(Dim,S,Names)(iota(Dim).map!(idx=>_value[idx] - rhs._value[idx]).array);
        }
        static if(op == "*"){
            return Vec!(Dim,S,Names)(iota(Dim).map!(idx=>_value[idx] * rhs._value[idx]).array);
        }
        static if(op == "/"){
            return Vec!(Dim,S,Names)(iota(Dim).map!(idx=>_value[idx] / rhs._value[idx]).array);
        }
    }
    auto opBinary(string op,R)(const R[Dim] rhs) const 
    {
        alias S = CommonType!(T,R);
        static if(op == "%"){
            return iota(Dim).map!(idx=>_value[idx]*rhs[idx]).reduce!"a+b";
        }
        static if(op == "+"){
            return Vec!(Dim,S,Names)(iota(Dim).map!(idx=>_value[idx] + rhs[idx]).array);
        }
        static if(op == "-"){
            return Vec!(Dim,S,Names)(iota(Dim).map!(idx=>_value[idx] - rhs[idx]).array);
        }
        static if(op == "*"){
            return Vec!(Dim,S,Names)(iota(Dim).map!(idx=>_value[idx] * rhs[idx]).array);
        }
        static if(op == "/"){
            return Vec!(Dim,S,Names)(iota(Dim).map!(idx=>_value[idx] / rhs[idx]).array);
        }
    }
    auto opBinary(string op, R)(const R rhs) const if(isNumeric!R)
    {
        alias S = CommonType!(T,R);
        static if(op == "+"){
            return Vec!(Dim,S,Names)(_value.dup.map!(a=>a + rhs).array);
        }
        static if(op == "-"){
            return Vec!(Dim,S,Names)(_value.dup.map!(a=>a - rhs).array);
        }
        static if(op == "*"){
            return Vec!(Dim,S,Names)(_value.dup.map!(a=>a * rhs).array);
        }
        static if(op == "/"){
            return Vec!(Dim,S,Names)(_value.dup.map!(a=>a / rhs).array);
        }
    }

    auto opBinaryRight(string op, R)(const R lhs) const if(isNumeric!R)
    {
        static if(op == "+"){
            return this + lhs;
        }
        static if(op == "-"){
            return this - lhs;
        }
        static if(op == "*"){
            return this * lhs;
        }
    }

    double norm2()
    {
        return this%this;
    }
    double norm()
    {
        return norm2.sqrt;
    }
    auto unit()
    {
        alias S = CommonType!(T,double);
        return norm2 == 0 ? Vec!(Dim,S,Names)((cast(S)0.).repeat(Dim).array) : this/norm;
    }
    auto opUnary(string op)() if(op=="-")
    {
        return 0-this;
    }
    static if(Dim == 2)
    {
        T cross(N...)(const Vec!(2,T,N) rhs)
        {
            return _value[0]*rhs[1]-_value[1]*rhs[0];
        }
    }
}