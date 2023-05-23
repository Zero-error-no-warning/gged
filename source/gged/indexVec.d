module ggeD.indexVec;
import ggeD;
import std;


private alias MakeSerialIndex(X) =  SerialIndex!X;

package(ggeD):
struct IndexVec(IndexTypes...) 
{

	alias SrialIndexes = staticMap!(MakeSerialIndex,IndexTypes);
	SrialIndexes idx;
	alias Dim = Alias!(IndexTypes.length);
	alias Rank = Dim;
    alias idx this;
	@nogc size_t[Dim] unit(ulong n)
	{
		size_t[Dim] vec = 0;
		vec[n] = 1;
		return vec;
	}
	alias tupleof = idx;
    @nogc auto opBinary(string op,OtherIndexTypes...)(IndexVec!(OtherIndexTypes) rhs) if(OtherIndexTypes.length == Dim)
    {
        SrialIndexes result = idx;
        foreach(i; 0 .. Dim)
        {
            result[i] = mixin("idx[i]" ~ op ~ "rhs[i]");
        }
        return IndexVec!(IndexTypes)(result);
    }
    @nogc auto opBinary(string op,N)(N[Dim] rhs)
    {
        SrialIndexes result = idx;
        static foreach(i; 0 .. Dim)
        {
            result[i] = mixin("idx[i]" ~ op ~ "rhs[i]");
        }
        return IndexVec!(IndexTypes)(result);
    }
    @nogc auto opBinaryRight(string op,N)(N[Dim] lhs)
    {
        SrialIndexes result = idx;
        static foreach(i; 0 .. Dim)
        {
            result[i] = mixin("lhs[i]" ~ op ~ "idx[i]");
        }
        return IndexVec!(IndexTypes)(result);
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
        return "["~instantWrite!", "(idx)~"]";
    }
	@nogc auto clamp()
	{
        SrialIndexes result = idx;
		static foreach(i;0..Dim)
		{
			result[i] = idx[i].clamp;
		}
		return IndexVec!(IndexTypes)(result);
	}
	@nogc auto loop()
	{
        SrialIndexes result = idx;
		static foreach(i;0..Dim)
		{
			result[i] = idx[i].loop;
		}
		return IndexVec!(IndexTypes)(result);
	}
}
        
struct SerialIndex(IndexType = int)
{
	this(T)(T f,size_t len_,IndexType offset = 0)
	{
		_idx = cast(IndexType)f;
        _len = len_;
	}
	@nogc const IndexType max()
	{
		return cast(IndexType)(_len+_offset);
	}
	@nogc const size_t len()
	{
		return _len;
	}
	
	IndexType _idx ;
    size_t _len;
	IndexType _offset = 0;
	alias _idx this;
	@nogc void idx(IndexType i){
		_idx = i;
	}
	@nogc auto idx()
	{
		return _idx;
	}
	@nogc auto opAssign(T)(T value)
	{
		_idx = cast(IndexType)value;
	}
	@nogc SerialIndex opUnary(string op)()
	{
		auto r = SerialIndex(mixin(op~"_idx"),_len,_offset);
		return r;
	}
	@nogc SerialIndex opBinary(string op,T)(T value)
	{
		auto r = SerialIndex(mixin("_idx"~op~"value"),_len,_offset);
		return r;
	}
	@nogc SerialIndex opBinaryRight(string op,T)(T value)
	{
		auto r = SerialIndex(mixin("value"~op~"_idx"),_len,_offset);
		return r;
	}
	@nogc SerialIndex clamp()
	{
		auto r = SerialIndex(_idx < 0 ? 0 : _idx > max ? max : _idx,_len,_offset);
		return r;
	}
	@nogc SerialIndex loop()
	{
		auto r = SerialIndex(_idx < 0 ? _len+_idx : _idx > max ? _idx-_len : _idx,_len,_offset);
		return r;
	}
    string toString()
    {
        return _idx.to!string;
    }
                    
}