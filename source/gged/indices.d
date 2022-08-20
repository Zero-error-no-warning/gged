module ggeD.indices;
import ggeD;
import std.conv;
import std.array;
import std.algorithm : map;

import std.meta;

struct Indices(ulong Dim,Idx) if (is(Idx == Index) || is(Idx == SerialIndex))
{
    Idx[Dim] _value;
    ref auto opIndex(ulong dim){
        return _value[dim];
    }
    auto opBinary(string op,T)(T[Dim] rhs){
		auto result = Indices!(Dim,Idx)(_value);
		static foreach (d;0..Dim){{
			mixin("result._value[d] " ~op~"= rhs[d];");
		}}
        return result;
    }
    auto opBinary(string op,T)(T[] rhs){
		auto result = Indices!(Dim,Idx)(_value);
		static foreach (d;0..Dim){{
			mixin("result._value[d] " ~op~"= rhs[d];");
		}}
        return result;
    }
    auto opBinary(string op,T)(Indices!(T,Dim) rhs){
		auto result = Indices!(Dim,Idx)(_value);
		static foreach (d;0..Dim){{
			mixin("result._value[d] " ~op~"= rhs._value[d];");
		}}
        return result;
    }
    auto opBinaryRight(string op,T)(T[Dim] lhs){
		auto result = Indices!(Dim,Idx)(lhs);
		static foreach (d;0..Dim){{
			mixin("result._value[d] " ~op~"= _value[d];");
		}}
        return result;
    }
    auto opBinary(string op,T)(T[] lhs){
		auto result = Indices!(Dim,Idx)(cast(T[Dim])lhs);
		static foreach (d;0..Dim){{
			mixin("result._value[d] " ~op~"= _value[d];");
		}}
        return result;
    }

    auto opUnary(string op)() if(op=="-"){
		auto result = Indices!(Dim,Idx)(_value);
		static foreach (d;0..Dim){{
			mixin("result._value[d] =" ~op~" result._value[d];");
		}}
        return result;
    }

	string toString() const @safe pure{
		return _value.array.map!(a=>a._idx).array.to!string;
	}

    auto loop(T)(Indices!(Dim,T) idx){
        auto result =  Indices!(Dim,Idx)(_value);
        static foreach(d; 0.. Dim){{
            result._value[d] =result._value[d].loop(idx._value[d]);
        }}
		return result;
    }
    auto loop(){
        auto result =  Indices!(Dim,Idx)(_value);
        static foreach(d; 0.. Dim){{
            result._value[d] =result._value[d].loop();
        }}
		return result;
    }
    auto clamp(T)(Indices!(Dim,T) idx){
        auto result =  Indices!(Dim,Idx)(_value);
        static foreach(d; 0.. Dim){{
            result._value[d] = result._value[d].clamp(idx._value[d]);
        }}
		return result;
    }
    auto clamp(){
        auto result =  Indices!(Dim,Idx)(_value);
        static foreach(d; 0.. Dim){{
            result._value[d] = result._value[d].clamp();
        }}
		return result;
    }
    auto base(ulong n)
    {
        auto r = Indices!(Dim,Idx)(Repeat!(Dim,Idx(0)));
        static foreach(d;0..Dim)
        {{
            r[d].tupleof = _value[d].tupleof;
            r[d]._idx = n == d ? 1 : 0;
        }}
        return r;
    }
    this(X...)(X value_) if(X.length == Dim)
    {
        static foreach(i;0..Dim)
        {
            _value[i] = Idx(value_[i]);
        }
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

    this(X)(const X[Dim] value_)
    {
        static foreach(i;0..Dim)
        {
            _value[i] = value_[i];
        }
    }

}

package(ggeD) 
struct SerialIndex
{
	this(T)(T f)
	{
		_idx = cast(long)f;
	}
	const ulong max()
	{
		return _len-1;
	}
	const ulong len()
	{
		return _len;
	}
	
	long _idx = long.max;
	alias _idx this;
	
	package(ggeD)  ulong _len;
	void idx(ulong i){
		_once = i != _idx;
		_idx = i;
	}
	bool _once = true;
	bool _last = false;
	const bool once()
	{
		return _once;
	}			
	const bool last()
	{
		return _last;
	}			
	auto opAssign(T)(T value)
	{
		_idx = cast(ulong)value;
	}
	SerialIndex opUnary(string op)()
	{
		auto r = SerialIndex(mixin(op~"_idx"));
		r._len = _len;
		return r;
	}
	SerialIndex opBinary(string op,T)(T value)
	{
		auto r = SerialIndex(mixin("_idx"~op~"value"));
		r._len = _len;
		return r;
	}
	SerialIndex opBinaryRight(string op,T)(T value)
	{
		auto r = SerialIndex(mixin("value"~op~"_idx"));
		r._len = _len;
		return r;
	}
	SerialIndex clamp(Idx)(Idx value) if (is(Idx == Index) || is(Idx == SerialIndex))
	{
		auto r = SerialIndex(_idx < 0 ? 0 : _idx > value.max ? value.max : _idx);
		r._len = _len;
		return r;
	}
	SerialIndex loop(Idx)(Idx value) if (is(Idx == Index) || is(Idx == SerialIndex))
	{
		auto r = SerialIndex(_idx < 0 ? value._len+_idx : _idx > value.max ? _idx-value._len : _idx);
		r._len = _len;
		return r;
	}
	SerialIndex clamp()
	{
		auto r = SerialIndex(_idx < 0 ? 0 : _idx > max ? max : _idx);
		r._len = _len;
		return r;
	}
	SerialIndex loop()
	{
		auto r = SerialIndex(_idx < 0 ? _len+_idx : _idx > max ? _idx-_len : _idx);
		r._len = _len;
		return r;
	}
}

package(ggeD) 
struct Index
{
	long _idx =0;
	alias _idx this;
	
	package(ggeD)  ulong _len =0;
	const ulong max()
	{
		return _len-1;
	}
	const ulong len()
	{
		return _len;
	}
	this(T)(T f)
	{
		_idx = cast(long)f;
	}
	auto opAssign(T)(T value)
	{
		_idx = cast(long)value;
	}
	Index opUnary(string op)()
	{
		auto r = Index(mixin(op~"_idx"));
		r._len = _len;
		return r;
	}
	Index opBinary(string op,T)(T value)
	{
		auto r = Index(mixin("_idx"~op~"value"));
		r._len = _len;
		return r;
	}
	Index opBinaryRight(string op,T)(T value)
	{
		auto r = Index(mixin("value"~op~"_idx"));
		r._len = _len;
		return r;
	}
	Index clamp(Idx)(Idx value) if (is(Idx == Index) || is(Idx == SerialIndex))
	{
		auto r = Index(_idx < 0 ? 0 : _idx > value.max ? value.max : _idx);
		r._len = _len;
		return r;
	}
	Index loop(Idx)(Idx value) if (is(Idx == Index) || is(Idx == SerialIndex))
	{
		auto r = Index(value < 0 ? value._len+value : value > value.max ? value-value._len : value);
		r._len = _len;
		return r;
	}
	Index clamp()
	{
		auto r = Index(_idx < 0 ? 0 : _idx > max ? max : _idx);
		r._len = _len;
		return r;
	}
	Index loop()
	{
		auto r = Index(_idx < 0 ? _len+_idx : _idx > max ? _idx-_len : _idx);
		r._len = _len;
		return r;
	}
}
import std.traits;
bool isIndex(X)(){
	static if(!__traits(isTemplate,X)) 
		return is(X == Index) ||  is(X == SerialIndex) || isIntegral!X ;
	else
		return  __traits(isSame,TemplateOf!X ,Vec) && isIntegral!((TemplateArgsOf!X)[1]);
} 
