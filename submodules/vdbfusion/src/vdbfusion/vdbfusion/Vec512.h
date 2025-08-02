#ifndef VEC512_H
#define VEC512_H

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>

template <class T>
class Vec512
{
public:
    static const size_t SIZE = 512;

    //-------------------
    // Access to elements
    //-------------------

    std::vector<T> data;

    T& operator[](size_t i) {
        return data[i];
    }

    const T& operator[](size_t i) const {
        return data[i];
    }

    //-------------
    // Constructors
    //-------------

    Vec512() : data(SIZE, T(0)) {}                        // no initialization
    explicit Vec512(T a) : data(SIZE, a) {}               // (a a a ... a)
    Vec512(const std::vector<T>& v) : data(v) {}          // (v[0] v[1] ... v[511])

    //---------------------------------
    // Copy constructors and assignment
    //---------------------------------

    Vec512(const Vec512& v) : data(v.data) {}
    template <class S> Vec512(const Vec512<S>& v) {
        data.resize(SIZE);
        for (size_t i = 0; i < SIZE; ++i) {
            data[i] = static_cast<T>(v[i]);
        }
    }

    const Vec512& operator=(const Vec512& v) {
        data = v.data;
        return *this;
    }

    //------------
    // Destructor
    //------------

    ~Vec512() = default;

    //----------------------
    // Compatibility with Sb
    //----------------------

    template <class S>
    void setValue(const std::vector<S>& v) {
        for (size_t i = 0; i < SIZE; ++i) {
            data[i] = static_cast<T>(v[i]);
        }
    }

    template <class S>
    void getValue(std::vector<S>& v) const {
        v.resize(SIZE);
        for (size_t i = 0; i < SIZE; ++i) {
            v[i] = static_cast<S>(data[i]);
        }
    }

    T* getValue() {
        return data.data();
    }

    const T* getValue() const {
        return data.data();
    }

    //---------
    // Equality
    //---------

    template <class S>
    bool operator==(const Vec512<S>& v) const {
        return data == v.data;
    }

    template <class S>
    bool operator!=(const Vec512<S>& v) const {
        return data != v.data;
    }

    //-----------------------------------------------------------------------
    // Compare two vectors and test if they are "approximately equal":
    //
    // equalWithAbsError (v, e)
    //
    //      Returns true if the coefficients of this and v are the same with
    //      an absolute error of no more than e, i.e., for all i
    //
    //      abs (this[i] - v[i]) <= e
    //
    // equalWithRelError (v, e)
    //
    //      Returns true if the coefficients of this and v are the same with
    //      a relative error of no more than e, i.e., for all i
    //
    //      abs (this[i] - v[i]) <= e * abs (this[i])
    //-----------------------------------------------------------------------

    bool equalWithAbsError(const Vec512<T>& v, T e) const {
        for (size_t i = 0; i < SIZE; ++i) {
            if (std::abs(data[i] - v[i]) > e) {
                return false;
            }
        }
        return true;
    }

    bool equalWithRelError(const Vec512<T>& v, T e) const {
        for (size_t i = 0; i < SIZE; ++i) {
            if (std::abs(data[i] - v[i]) > e * std::abs(data[i])) {
                return false;
            }
        }
        return true;
    }

    //------------
    // Dot product
    //------------

    T dot(const Vec512& v) const {
        T result = T(0);
        for (size_t i = 0; i < SIZE; ++i) {
            result += data[i] * v[i];
        }
        return result;
    }

    T operator^(const Vec512& v) const {
        return dot(v);
    }

    //------------------------
    // Component-wise addition
    //------------------------

    const Vec512& operator+=(const Vec512& v) {
        for (size_t i = 0; i < SIZE; ++i) {
            data[i] += v[i];
        }
        return *this;
    }

    Vec512 operator+(const Vec512& v) const {
        Vec512 result(*this);
        result += v;
        return result;
    }

    //---------------------------
    // Component-wise subtraction
    //---------------------------

    const Vec512& operator-=(const Vec512& v) {
        for (size_t i = 0; i < SIZE; ++i) {
            data[i] -= v[i];
        }
        return *this;
    }

    Vec512 operator-(const Vec512& v) const {
        Vec512 result(*this);
        result -= v;
        return result;
    }

    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    Vec512 operator-() const {
        Vec512 result(*this);
        for (size_t i = 0; i < SIZE; ++i) {
            result[i] = -data[i];
        }
        return result;
    }

    const Vec512& negate() {
        for (size_t i = 0; i < SIZE; ++i) {
            data[i] = -data[i];
        }
        return *this;
    }

    //------------------------------
    // Component-wise multiplication
    //------------------------------

    const Vec512& operator*=(const Vec512& v) {
        for (size_t i = 0; i < SIZE; ++i) {
            data[i] *= v[i];
        }
        return *this;
    }

    const Vec512& operator*=(T a) {
        for (size_t i = 0; i < SIZE; ++i) {
            data[i] *= a;
        }
        return *this;
    }

    Vec512 operator*(const Vec512& v) const {
        Vec512 result(*this);
        result *= v;
        return result;
    }

    Vec512 operator*(T a) const {
        Vec512 result(*this);
        result *= a;
        return result;
    }

    //------------------------
    // Component-wise division
    //------------------------

    const Vec512& operator/=(const Vec512& v) {
        for (size_t i = 0; i < SIZE; ++i) {
            data[i] /= v[i];
        }
        return *this;
    }

    const Vec512& operator/=(T a) {
        for (size_t i = 0; i < SIZE; ++i) {
            data[i] /= a;
        }
        return *this;
    }

    Vec512 operator/(const Vec512& v) const {
        Vec512 result(*this);
        result /= v;
        return result;
    }

    Vec512 operator/(T a) const {
        Vec512 result(*this);
        result /= a;
        return result;
    }

    //----------------------------------------------------------------
    // Length and normalization:  If v.length() is 0.0, v.normalize()
    // and v.normalized() produce a null vector; v.normalizeExc() and
    // v.normalizedExc() throw a NullVecExc.
    // v.normalizeNonNull() and v.normalizedNonNull() are slightly
    // faster than the other normalization routines, but if v.length()
    // is 0.0, the result is undefined.
    //----------------------------------------------------------------

    T length() const {
        return std::sqrt(length2());
    }

    T length2() const {
        T result = T(0);
        for (size_t i = 0; i < SIZE; ++i) {
            result += data[i] * data[i];
        }
        return result;
    }

    const Vec512& normalize() {
        T len = length();
        if (len != T(0)) {
            *this /= len;
        }
        return *this;
    }

    const Vec512& normalizeExc() {
        T len = length();
        if (len == T(0)) {
            throw std::runtime_error("Null vector normalization exception");
        }
        *this /= len;
        return *this;
    }

    const Vec512& normalizeNonNull() {
        *this /= length();
        return *this;
    }

    Vec512<T> normalized() const {
        Vec512 result(*this);
        result.normalize();
        return result;
    }

    Vec512<T> normalizedExc() const {
        Vec512 result(*this);
        result.normalizeExc();
        return result;
    }

    Vec512<T> normalizedNonNull() const {
        Vec512 result(*this);
        result.normalizeNonNull();
        return result;
    }

    //--------------------------------------------------------
    // Number of dimensions, i.e. number of elements in a Vec512
    //--------------------------------------------------------

    static unsigned int dimensions() { return SIZE; }

    //-------------------------------------------------
    // Limitations of type T (see also class limits<T>)
    //-------------------------------------------------

    static T baseTypeMin() { return std::numeric_limits<T>::min(); }
    static T baseTypeMax() { return std::numeric_limits<T>::max(); }
    static T baseTypeSmallest() { return std::numeric_limits<T>::min(); }
    static T baseTypeEpsilon() { return std::numeric_limits<T>::epsilon(); }

    //--------------------------------------------------------------
    // Base type -- in templates, which accept a parameter, V, which
    // could be either a Vec2<T>, a Vec3<T>, or a Vec4<T> you can 
    // refer to T as V::BaseType
    //--------------------------------------------------------------

    typedef T BaseType;

private:
    T lengthTiny() const {
        return std::sqrt(std::numeric_limits<T>::min());
    }
};

#endif // VEC512_H