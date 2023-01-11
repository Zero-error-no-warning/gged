module ggeD.indexVec;
import ggeD;
import std;


struct IndexVec(size_t dim)
{
	SerialIndex[dim] idx;
	alias Dim = dim;
    alias idx this;
    void opBinary(string op,N)(N[dim] rhs)
    {
        SerialIndex[dim] result;
        foreach(i; 0 .. dim)
        {
            result[i] = mixin("idx[i]" ~ op ~ "rhs[i]");
        }
        return IndexVec!dim(result);
    }
    void opBinaryRight(string op,N)(N[dim] lhs)
    {
        SerialIndex[dim] result;
        foreach(i; 0 .. dim)
        {
            result[i] = mixin("lhs[i]" ~ op ~ "idx[i]");
        }
        return IndexVec!dim(result);
    }
    string toString()
    {
        string instantWrite(string sep="",Arg...)(Arg arg)
        {
            string result;
            foreach(i,v;arg)
            {
                result ~= v.to!string;
                if(i!=arg.length-1) result~= sep;
            }
            return result;
        }
        return instantWrite!", "(idx);
    }
}
        
struct SerialIndex
{
	alias Dim = Alias!1;
	this(T)(T f,ulong len_)
	{
		_idx = cast(long)f;
        _len = len_;
	}
	const ulong max()
	{
		return _len-1;
	}
	const ulong len()
	{
		return _len;
	}
	
	long _idx ;
    ulong _len;
	alias _idx this;
	void idx(ulong i){
		_idx = i;
	}
	auto idx()
	{
		return _idx;
	}
	auto opAssign(T)(T value)
	{
		_idx = cast(ulong)value;
	}
	SerialIndex opUnary(string op)()
	{
		auto r = SerialIndex(mixin(op~"_idx"),_len);
		return r;
	}
	SerialIndex opBinary(string op,T)(T value)
	{
		auto r = SerialIndex(mixin("_idx"~op~"value"),_len);
		return r;
	}
	SerialIndex opBinaryRight(string op,T)(T value)
	{
		auto r = SerialIndex(mixin("value"~op~"_idx"),_len);
		return r;
	}
	SerialIndex clamp()
	{
		auto r = SerialIndex(_idx < 0 ? 0 : _idx > max ? max : _idx,_len);
		return r;
	}
	SerialIndex loop()
	{
		auto r = SerialIndex(_idx < 0 ? _len+_idx : _idx > max ? _idx-_len : _idx,_len);
		return r;
	}
    string toString()
    {
        return _idx.to!string;
    }
                    
}