/*
Copyright (c) 2022 Zenw
Released under the MIT license
https://opensource.org/licenses/mit-license.php
*/

module ggeD.ggeD;
import std;
public import ggeD.vec;

/// create empty gged array 
/// Params:
///   N = shape of array
auto gged(T,X...)(X N) if(allSatisfy!(isIndex,X))
{
	return new Gged!(T,X.length)(N);
} 

/// 
/// Params:
///   array = source 1-dim array.
///   N =  shape of array
auto gged(T,X...)(T[] array,X N)  if(allSatisfy!(isIndex,X))
{
	return new Gged!(T,X.length)(array,N);
}

/// 
/// Params:
///   N =  shape of array
auto gged(T,size_t X,L)(L[X] N) if(isIndex!L)
{
	return new Gged!(T,X)(N);
}

/// 
/// Params:
///   array = source 1-dim array.
///   N =  shape of array
auto gged(T,size_t X,L)(T[] array,L[X] N) if(isIndex!L)
{
	return new Gged!(T,X)(array,N);
}

/// Gged class
class Gged(T,ulong Rank)
{
	alias TYPE = T;
	alias RANK = Alias!(Rank);
	private T[] _array;

	/// 
	/// Returns: elements as 1-dim array.
	T[] elements()
	{
		return _array.indexed(iota(_AllLength).filter!((i){
			Repeat!(Rank,size_t) idx2xyz;
			bool inRange = true;
			static foreach(n ; 0..Rank)
			{{
				idx2xyz[n] = (i /  _step[n])  % _rawN[n] ;
				inRange &= _since[n] <=idx2xyz[n]&& idx2xyz[n]<_until[n];
			}}
			return inRange;
		})).array;
	}

	/// 
	/// Returns: dupplication of gged.
	typeof(this) dup(Flag!"shape" OnlyShape = No.shape)()
	{
		if(OnlyShape) return new Gged!(T,Rank)(_rawN.dup);
		else return new Gged!(T,Rank)(_array.dup,_rawN.dup);
	}

	private immutable ulong _AllLength;
	private immutable ulong[Rank] _step;
	private immutable ulong[Rank] _rawN;
	private ulong[Rank] _since;
	private ulong[Rank] _until;
	TaskPool customPool;

	/// 
	/// Returns: shape of gged array.
	ulong[Rank] shape()
	{
		return _rawN[];
	}

	/// 
	/// Params:
	///   array = source 1-dim array.
	this(T[] array,ulong[] N...) 
	{
		assert(N.length == Rank,format("number of args must be same as Rank: %s , %s" , N, Rank));
		_rawN = cast(immutable)N;
		_AllLength = N.reduce!((a,b)=>a*b);
		_array = array;

		ulong[Rank] step;
		ulong n = 1;

		step[0] = 1;
		static foreach(i;1..Rank)
		{{
			n *= _rawN[i-1];
			step[i] = n;
		}}
		static foreach(i;0..Rank)
		{{
			_until[i] = _rawN[i];
			_since[i] = 0;
		}}
		_step = cast(immutable)(step);
		assert(_AllLength == elements.length, "array length and shape are not corresponded");
	}
		
		

	/// 
	/// Params:
	///   array = source 1-dim array.
	this(ulong[] N...) 
	{
		assert(N.length == Rank,format("number of args must be same as Rank: %s , %s" , N, Rank));
		_rawN = cast(immutable)N;
		_AllLength = N.reduce!((a,b)=>a*b);
		_array = new T[](_AllLength);

		ulong[Rank] step;
		ulong n = 1;

		step[0] = 1;
		static foreach(i;1..Rank)
		{{
			n *= _rawN[i-1];
			step[i] = n;
		}}
		static foreach(i;0..Rank)
		{{
			_until[i] = _rawN[i];
			_since[i] = 0;
		}}
		_step = cast(immutable)(step);
	}

	scope auto Serial()
	{
		alias TypeSerialIndex = Repeat!(Rank,SerialIndex);
		return new class 
		{
			int opApply(int delegate(TypeSerialIndex) fun) 
			{
				scope AliasSeq!TypeSerialIndex idx2xyz;
				static foreach(n ; 0..Rank)
				{{
					idx2xyz[n] = SerialIndex();
				}}
				foreach(i; iota(_AllLength))
				{
					bool inRange = true;
					static foreach(n ; 0..Rank)
					{{
						idx2xyz[n].idx = (i /  _step[n])  % _rawN[n] ;
						idx2xyz[n]._last = idx2xyz[n]._idx == _until[n] - 1; 
						idx2xyz[n]._max = _until[n]-_since[n];
						inRange &= _since[n] <=idx2xyz[n]._idx && idx2xyz[n]._idx<_until[n];
						idx2xyz[n]._idx -= _since[n] ;
					}}
					if(inRange) fun(idx2xyz);
				}
				return 1;
			}
			static if(Rank>1)  int opApply(int delegate(Vec!(Rank,SerialIndex)) fun) 
			{
				scope Vec!(Rank,SerialIndex) idx2xyz;
				static foreach(n ; 0..Rank)
				{{
					idx2xyz[n] = SerialIndex();
				}}
				foreach(i; iota(_AllLength))
				{
					bool inRange = true;
					static foreach(n ; 0..Rank)
					{{
						idx2xyz[n].idx = (i /  _step[n])  % _rawN[n] ;
						idx2xyz[n]._last = idx2xyz[n]._idx == _until[n] - 1; 
						idx2xyz[n]._max = _until[n]-_since[n];
						inRange &= _since[n] <=idx2xyz[n]._idx && idx2xyz[n]._idx<_until[n];
						idx2xyz[n]._idx -= _since[n] ;
					}}
					if(inRange) fun(idx2xyz);
				}
				return 1;
			}
		};
	}
	scope Elemental()
	{
		return new class 
		{
			int opApply(int delegate(ref T) fun) 
			{
				foreach(i,ref elem;parallel(_array))
				{
					Repeat!(Rank,size_t) idx2xyz;
					bool inRange = true;
					static foreach(n ; 0..Rank)
					{{
						idx2xyz[n] = (i /  _step[n])  % _rawN[n] ;
						inRange &= _since[n] <=idx2xyz[n]&& idx2xyz[n]<_until[n];
					}}
					if(inRange) fun(elem);
				}
				return 1;
			}
		};
	}
    int opApply(int delegate(Repeat!(Rank,Index)) fun) 
    {
		foreach(i; parallel(iota(_AllLength)))
		{
			Repeat!(Rank,Index) idx2xyz;
			bool inRange = true;
			static foreach(n ; 0..Rank)
			{{
				idx2xyz[n]._idx = (i /  _step[n])  % _rawN[n];
				idx2xyz[n]._max = _until[n]-_since[n];
				inRange &= _since[n]<=idx2xyz[n]._idx && idx2xyz[n]._idx<_until[n];
				idx2xyz[n]._idx -= _since[n] ;
			}}
			if(inRange) fun(idx2xyz);
		}
		return 1;
    }

    static if(Rank>1) int opApply(int delegate(Vec!(Rank,Index)) fun) 
    {
		foreach(i; parallel(iota(_AllLength)))
		{
			Vec!(Rank,Index) idx2xyz;
			bool inRange = true;
			static foreach(n ; 0..Rank)
			{{
				idx2xyz[n]._idx = (i /  _step[n])  % _rawN[n];
				idx2xyz[n]._max = _until[n]-_since[n];
				inRange &= _since[n]<=idx2xyz[n]._idx && idx2xyz[n]._idx<_until[n];
				idx2xyz[n]._idx -= _since[n] ;
			}}
			if(inRange) fun(idx2xyz);
		}
		return 1;
    }
	ref T opIndex(X)(Vec!(Rank,X) arg) @nogc if(isIndex!X)
	{
		ulong n = 0;
		static foreach(i; 0 .. Rank)
		{{
			n += (_since[i] +  arg[i])*_step[i];
		}}
		return _array[n];
	}
	ref T opIndex(X...)(X arg) @nogc if(allSatisfy!(isIndex,X))
	{
		ulong n = 0;
		static foreach(i; 0 .. Rank)
		{{
			n += (_since[i] +  arg[i])*_step[i];
		}}
		return _array[n];
	}
	auto opIndex(Flag!"Cut" cutted = No.Cut, X...)(X arg) if((X.length > 1 &&!allSatisfy!(isIndex,X))|| (X.length==1 && isArray!(X) ) )
	{
		template IndexOfnotIndex(ulong N,X...)
		{
			static if(X.length == 1)
				static if(!isIndex!X)
					alias IndexOfnotIndex =  AliasSeq!N;
				else
					alias IndexOfnotIndex = AliasSeq!();
			else
			{
				alias IndexOfnotIndex =  AliasSeq!(IndexOfnotIndex!(N,X[0]),IndexOfnotIndex!(N+1,X[1..$]));
			}
		}
		static if(cutted)
		{ 
			alias idx = IndexOfnotIndex!(0,X);
			auto f = opIndex!(No.Cut,X)(arg);
			auto g = new Gged!(T,Rank-Filter!(isIndex,X).length)(f.elements,indexed(shape.to!(ulong[]),[idx]).array);
			ulong n = 0;
			static foreach(i; 0 ..X.length)
			{{
				static if(isArray!(typeof(arg[i])))
				{
					g._since[n] = arg[i][0];
					g._until[n] = arg[i][1];
					n ++ ;
				}
				else
				{
				}
			}}
			return g;
		}
		else
		{
			auto g = gged!T(_array,shape);
			static foreach(i; 0 .. X.length)
			{{
				static if(isArray!(typeof(arg[i])))
				{
					g._since[i] = arg[i][0];
					g._until[i] = arg[i][1];
				}
				else
				{
					g._since[i] = arg[i];
					g._until[i] = arg[i]+1;
				}
			}}
			return g;
		}
	}
	static if(Rank == 1)
	{
		auto opSlice(X,Y)(X start, Y end) if(isIndex!X && isIndex!Y)
		{
			auto g = gged!T(_array,shape);
			g._since[0] = start;
			g._until[0] = end;
			return g;
		}
	}
	else
	{
		size_t[2] opSlice(size_t dim,X,Y)(X start, Y end) if(isIndex!X && isIndex!Y)
		{
			return [start, end];
		}
	}


	size_t opDollar(ulong rank)()
	{
		return _rawN[rank];
	}
	

	auto opSliceAssign(T)(T value)
	{
		_array[] = value;
	}
	
	auto opSliceAssign(Gged!(T,Rank) value) 
	in
	{
		assert(value._rawN[] == _rawN[]);
	}do
	{
		_array = value._array.dup;
	}


	scope t() 
	{
		alias realthis = this;
		return new class {
			
			auto opIndex(X)(Vec!(Rank,X) idx) @nogc
			{
				return new subGGeD!(T,Rank,X)(realthis,idx);
			}
			auto opIndex(size_t[2] se)
			{
				return new subGGeD!(T,Rank,size_t[2])(realthis,se);
			}
			scope opIndex(X...)(X idx) 
			{
				return new subGGeD!(T,Rank,X)(realthis,idx);
			}
			static if(__traits(hasMember,T,"opDispatch"))
			{
				ref auto opDispatch(string member)()
				{
					return new subGGeD!(T,Rank,member)(realthis);
				}
			}
			size_t[2] opSlice(size_t rank,X,Y)(X start, Y end) if(isIndex!X && isIndex!Y)
			{
				return [start, end];
			}
			size_t opDollar(ulong rank)()
			{
				static if(__traits(hasMember,T,"opDollar"))
				{
					return T.opDollar!rank;
				}
				else
				{
					return _array[0].length;
				}
			}
		};
	}
}

package(ggeD)
class subGGeD(T,ulong Rank,N...)
{
	Gged!(T,Rank) _gged;
	alias _gged this;
	static if(N.length > 1 || is(N[0] == size_t[2]))
	{
		N _N;
		this(Gged!(T,Rank) gged_,N N_) 
		{
			_gged = gged_;
			_N = N_;
		}
		static if(N.length == 1 && is(N[0]== size_t[2]))
		{
			ref auto opIndex(X...)(X arg) @nogc 
			{
				return _gged[arg][_N[0][0] .. _N[0][1]];
			}
		}
		else
		{
			ref auto opIndex(X...)(X arg) @nogc 
			{
				return _gged[arg][_N];
			}
			ref auto opIndex(X)(Vec!(Rank,X) arg) @nogc 
			{
				return _gged[arg][_N];
			}
		}
	}
	else static if(is(typeof(N[0])==string))
	{
		enum _N = N[0];
		this(Gged!(T,Rank) gged_) 
		{
			_gged = gged_;
		}
		static if( __traits(hasMember,T,"opDispatch"))
		ref auto opIndex(X...)(X arg) @nogc 
		{
			return _gged[arg].opDispatch!(_N);
		}
		ref auto opIndex(X)(Vec!(Rank,X) arg) @nogc 
		{
			return _gged[arg].opDispatch!(_N);
		}
	}
	
}

package(ggeD) 
struct Index
{
	ulong _idx =0;
	alias _idx this;
	ulong _max =0;
	const ulong max()
	{
		return _max;
	}
	this(ulong f)
	{
		_idx = f;
	}
	this(double f)
	{
		_idx = cast(ulong)f;
	}
	this(const(double) f)
	{
		_idx = cast(ulong)f;
	}
	auto opAssign(T)(T value)
	{
		_idx = cast(ulong)value;
	}
}

bool isIndex(X)(){
	static if(!__traits(isTemplate,X)) 
		return is(X == Index) ||  is(X == SerialIndex) || isIntegral!X ;
	else
		return  __traits(isSame,TemplateOf!X ,Vec) && isIntegral!((TemplateArgsOf!X)[1]);
} 

package(ggeD) 
struct SerialIndex
{
	this(ulong f)
	{
		_idx = f;
	}
	this(double f)
	{
		_idx = cast(ulong)f;
	}
	this(const(double) f)
	{
		_idx = cast(ulong)f;
	}
	ulong _idx = ulong.max;
	alias _idx this;
	ulong _max;
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
}