module ggeD.indexVec;
import ggeD;
import std;

package(ggeD):
struct IndexVec(size_t dim)
{
	SerialIndex[dim] idx;
	alias Dim = dim;
    alias idx this;
    @nogc void opBinary(string op,N)(N[dim] rhs)
    {
        SerialIndex[dim] result;
        foreach(i; 0 .. dim)
        {
            result[i] = mixin("idx[i]" ~ op ~ "rhs[i]");
        }
        return IndexVec!dim(result);
    }
    @nogc void opBinaryRight(string op,N)(N[dim] lhs)
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
	@nogc const ulong max()
	{
		return _len-1;
	}
	@nogc const ulong len()
	{
		return _len;
	}
	
	long _idx ;
    ulong _len;
	alias _idx this;
	@nogc void idx(ulong i){
		_idx = i;
	}
	@nogc auto idx()
	{
		return _idx;
	}
	@nogc auto opAssign(T)(T value)
	{
		_idx = cast(ulong)value;
	}
	@nogc SerialIndex opUnary(string op)()
	{
		auto r = SerialIndex(mixin(op~"_idx"),_len);
		return r;
	}
	@nogc SerialIndex opBinary(string op,T)(T value)
	{
		auto r = SerialIndex(mixin("_idx"~op~"value"),_len);
		return r;
	}
	@nogc SerialIndex opBinaryRight(string op,T)(T value)
	{
		auto r = SerialIndex(mixin("value"~op~"_idx"),_len);
		return r;
	}
	@nogc SerialIndex clamp()
	{
		auto r = SerialIndex(_idx < 0 ? 0 : _idx > max ? max : _idx,_len);
		return r;
	}
	@nogc SerialIndex loop()
	{
		auto r = SerialIndex(_idx < 0 ? _len+_idx : _idx > max ? _idx-_len : _idx,_len);
		return r;
	}
    string toString()
    {
        return _idx.to!string;
    }
                    
}