module ggeD.iterator;
import mir.qualifier;
import mir.ndslice;
import std.traits;

struct EinsumIterator(Iterator,SomeSlice,Node) if(is(SomeSlice == Slice!(T, Rank, kind),T,ulong Rank,SliceKind kind))
{
    Iterator _iterator;
    SomeSlice _slice;
    Node node_;

    static enum Rank = SomeSlice.N;

    private size_t[Rank] ijk(Iterator itr)
    {
        size_t num = cast(size_t)(itr - _slice._iterator);
        size_t[Rank] result;
        static foreach(i; 0..Rank)
        {
            result[i] = cast(size_t)(num/_slice._stride!i);
            num -= result[i]*_slice._stride!i;
        }
        return result;
    }
    auto lightConst()() const @property
    {
        return EinsumIterator!(LightConstOf!Iterator,SomeSlice, Node)(mir.qualifier.lightConst(_iterator),_slice,cast(Unqual!Node)node_);
    }
    auto lightImmutable()() immutable @property
    {
        return EinsumIterator!(LightImmutableOf!Iterator, SomeSlice, Node)(mir.qualifier.lightImmutable(_iterator),cast(Unqual!Node)_slice,node_);
    }
        import std.stdio;

    auto ref opUnary(string op : "*")()
    {
        return node_.calc(ijk(_iterator).tupleof);
    }

    auto ref opIndex(ptrdiff_t index) scope
    {
        return node_.calc(ijk(_iterator+index).tupleof);
    }
    static if (!__traits(compiles, &opIndex(ptrdiff_t.init)))
    {
        auto ref opIndexAssign(T)(auto ref T value, ptrdiff_t index) scope
        {
            return _iterator[index] = value;
        }

        auto ref opIndexUnary(string op)(ptrdiff_t index)
        {
            return mixin(op ~ "node_.calc(ijk(_iterator+index).tupleof)");
        }

        auto ref opIndexOpAssign(string op, T)(T value, ptrdiff_t index)
        {
            return mixin("_iterator[index]" ~ op ~ "= value");
        }
    }
    

    void opUnary(string op)() scope
        if (op == "--" || op == "++")
    { mixin(op ~ "_iterator;"); }

    void opOpAssign(string op)(ptrdiff_t index) scope
        if (op == "-" || op == "+")
    { mixin("_iterator " ~ op ~ "= index;"); }

    auto opBinary(string op)(ptrdiff_t index)
        if (op == "+" || op == "-")
    {
        auto ret = this;
        mixin(`ret ` ~ op ~ `= index;`);
        return ret;
    }

    ptrdiff_t opBinary(string op : "-")(scope ref const typeof(this) right) scope const
    { return this._iterator - right._iterator; }

    bool opEquals()(scope ref const typeof(this) right) scope const
    { return this._iterator == right._iterator; }

    ptrdiff_t opCmp()(scope ref const typeof(this) right) scope const
    {
        static if (isPointer!Iterator)
            return this._iterator - right._iterator;
        else
            return this._iterator.opCmp(right._iterator);
    }

}