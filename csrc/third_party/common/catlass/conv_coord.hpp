/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_COORD_HPP
#define CATLASS_CONV_COORD_HPP

#include "catlass/coord.hpp"

namespace Catlass {

/// Shape of conv3d operation
struct Conv3dParams {
public:
    typedef uint32_t Index;
    static constexpr uint32_t N0 = 16;
    using Fmap6HDShape = Coord<6, Index>;   // {batch, di, cin1, hi, wi, cin0}
    using FilterFracZ3DShape = Coord<7, Index>; // {kd, cin1, kh, kw, n1, n0, cin0}
    using Out6HDShape = Coord<6, Index>;   // {batch, do, cout1, ho, wo, cout0}
    using Strides = Coord<3, Index>;
    using Pads = Coord<3, Index>;
    using Dilations = Coord<3, Index>;
private:
    Fmap6HDShape fmap6HDShape_;
    FilterFracZ3DShape filterFracZ3DShape_;
    Out6HDShape out6HDShape_;
    Strides strides_;
    Pads pads_;
    Dilations dilations_;
    Index cout_;
public:
    CATLASS_HOST_DEVICE
    Conv3dParams(
        Index BATCH = 1,
        Index Di = 1,
        Index Cin1 = 1,
        Index Hi = 1,
        Index Wi = 1,
        Index C0 = 16,
        Index Kd = 1,
        Index Kh = 1,
        Index Kw = 1,
        Index N1 = 1,
        Index Do = 1,
        Index Ho = 1,
        Index Wo = 1,
        Index Cout1 = 1,
        Index Cout = 1,
        Index padHead = 0,
        Index padTop = 0,
        Index padLeft = 0,
        Index strideD = 1,
        Index strideH = 1,
        Index strideW = 1,
        Index dilationD = 1,
        Index dilationH = 1,
        Index dilationW = 1
    )
    : fmap6HDShape_(MakeCoord(BATCH, Di, Cin1, Hi, Wi, C0)),
      filterFracZ3DShape_(MakeCoord(Kd, Cin1, Kh, Kw, N1, N0, C0)),
      out6HDShape_(MakeCoord(BATCH, Do, Cout1, Ho, Wo, C0)),
      cout_(Cout),
      pads_(MakeCoord(padHead, padTop, padLeft)),
      strides_(MakeCoord(strideD, strideH, strideW)),
      dilations_(MakeCoord(dilationD, dilationH, dilationW))
    {}

    CATLASS_HOST_DEVICE
    static Conv3dParams MakeConvCoord(
        const uint32_t* fmapShape,
        const uint32_t* filterShape,
        const uint32_t* paddings,
        const uint32_t* strides,
        const uint32_t* dilations)
    {
        return Conv3dParams(
            fmapShape[0],
            fmapShape[1],
            fmapShape[2],
            fmapShape[3],
            fmapShape[4],
            fmapShape[5],
            filterShape[0],
            filterShape[1],
            filterShape[2],
            CeilDiv(filterShape[3], N0),
            (fmapShape[1] + paddings[0] * 2 - dilations[0] * (filterShape[0] - 1) - 1) / strides[0] + 1,  //Do
            (fmapShape[3] + paddings[1] * 2 - dilations[1] * (filterShape[1] - 1) - 1) / strides[1] + 1,  //Ho
            (fmapShape[4] + paddings[2] * 2 - dilations[2] * (filterShape[2] - 1) - 1) / strides[2] + 1,  //Wo
            CeilDiv(filterShape[3], fmapShape[5]),
            filterShape[3],
            paddings[0],
            paddings[1],
            paddings[2],
            strides[0],
            strides[1],
            strides[2],
            dilations[0],
            dilations[1],
            dilations[2]
        );
    }
    

    //fmapShape
    CATLASS_HOST_DEVICE
    Index const &batch() const { return fmap6HDShape_[0]; }
    CATLASS_HOST_DEVICE
    Index const &cin1() const { return fmap6HDShape_[2]; }
    CATLASS_HOST_DEVICE
    Index const &di() const { return fmap6HDShape_[1]; }
    CATLASS_HOST_DEVICE
    Index const &hi() const { return fmap6HDShape_[3]; }
    CATLASS_HOST_DEVICE
    Index const &wi() const { return fmap6HDShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const &cin0() const { return fmap6HDShape_[5]; }
    CATLASS_HOST_DEVICE
    Index const hiwi() const { return fmap6HDShape_[3] * fmap6HDShape_[4]; }

    //filterShape
    CATLASS_HOST_DEVICE
    Index const &kd() const { return filterFracZ3DShape_[0]; }
    CATLASS_HOST_DEVICE
    Index const &kh() const { return filterFracZ3DShape_[2]; }
    CATLASS_HOST_DEVICE
    Index const &kw() const { return filterFracZ3DShape_[3]; }
    CATLASS_HOST_DEVICE
    Index const khkw() const
    { 
        return filterFracZ3DShape_[2] * filterFracZ3DShape_[3];
    }
    CATLASS_HOST_DEVICE
    Index const kdc1khkw() const{
        return filterFracZ3DShape_[0] * filterFracZ3DShape_[1] * filterFracZ3DShape_[2] * filterFracZ3DShape_[3];
    }
    CATLASS_HOST_DEVICE
    Index const &n1() const { return filterFracZ3DShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const &n0() const { return filterFracZ3DShape_[5]; }

    //outShape
    CATLASS_HOST_DEVICE
    Index const &dout() const { return out6HDShape_[1]; }
    CATLASS_HOST_DEVICE
    Index const &ho() const { return out6HDShape_[3]; }
    CATLASS_HOST_DEVICE
    Index const &wo() const { return out6HDShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const &cout1() const { return out6HDShape_[2]; }
    CATLASS_HOST_DEVICE
    Index const &cout0() const { return out6HDShape_[5]; }
    CATLASS_HOST_DEVICE
    Index const &cout() const { return cout_; }

    /// paddings
    CATLASS_HOST_DEVICE
    Index const &padhead() const { return pads_[0]; }
    CATLASS_HOST_DEVICE
    Index const &padtail() const { return pads_[0]; }
    CATLASS_HOST_DEVICE
    Index const &padtop() const { return pads_[1]; }
    CATLASS_HOST_DEVICE
    Index const &padbottom() const { return pads_[1]; }
    CATLASS_HOST_DEVICE
    Index const &padleft() const { return pads_[2]; }
    CATLASS_HOST_DEVICE
    Index const &padright() const { return pads_[2]; }

    /// strideSize
    CATLASS_HOST_DEVICE
    Index const &sD() const { return strides_[0]; }
    CATLASS_HOST_DEVICE
    Index const &sH() const { return strides_[1]; }
    CATLASS_HOST_DEVICE
    Index const &sW() const { return strides_[2]; }

    /// dilationSize
    CATLASS_HOST_DEVICE
    Index const &dD() const { return dilations_[0]; }
    CATLASS_HOST_DEVICE
    Index const dilatedKernelD() const { return 1 + (filterFracZ3DShape_[0] - 1) * dilations_[0]; }
    CATLASS_HOST_DEVICE
    Index const &dH() const { return dilations_[1]; }
    CATLASS_HOST_DEVICE
    Index const dilatedKernelH() const { return 1 + (filterFracZ3DShape_[2] - 1) * dilations_[1]; }
    CATLASS_HOST_DEVICE
    Index const &dW() const { return dilations_[2]; }
    CATLASS_HOST_DEVICE
    Index const dilatedKernelW() const { return 1 + (filterFracZ3DShape_[3] - 1) * dilations_[2]; }
    
    ///// used in block
    CATLASS_HOST_DEVICE
    Index const howo() const { return out6HDShape_[3] * out6HDShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const alignCout() const { return out6HDShape_[2] * out6HDShape_[5]; }
    CATLASS_HOST_DEVICE
    Index const wicin0() const { return fmap6HDShape_[4] * fmap6HDShape_[5]; }
    CATLASS_HOST_DEVICE
    Index const khkwcin0() const { return filterFracZ3DShape_[2] * filterFracZ3DShape_[3] * filterFracZ3DShape_[6]; }
    CATLASS_HOST_DEVICE
    Index const alignCinKhKwKd() const
    { 
        return filterFracZ3DShape_[0] * filterFracZ3DShape_[1] * filterFracZ3DShape_[2] * filterFracZ3DShape_[3] * filterFracZ3DShape_[6];
    }
    CATLASS_HOST_DEVICE
    Index const kdcin1() const { return filterFracZ3DShape_[0] * filterFracZ3DShape_[1]; }
    CATLASS_HOST_DEVICE
    Index const fmapOneBatchSize() const
    {
        return fmap6HDShape_[1] * fmap6HDShape_[2] * fmap6HDShape_[3] * fmap6HDShape_[4] * fmap6HDShape_[5];
    }
    CATLASS_HOST_DEVICE
    Index const outputOneBatchSize() const
    {
        return out6HDShape_[1] * out6HDShape_[2] * out6HDShape_[3] * out6HDShape_[4] * out6HDShape_[5];
    }
};

template <
    uint32_t noCnt_ = 1,
    uint32_t doCnt_ = 1,
    uint32_t co1Cnt_ = 1,
    uint32_t howoCnt_ = 1
>
struct ConvCoreShape {
    static uint32_t const noCnt = noCnt_;
    static uint32_t const doCnt = doCnt_;
    static uint32_t const co1Cnt = co1Cnt_;
    static uint32_t const howoCnt = howoCnt_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<4> ToCoord()
    {
        return MakeCoord(noCnt, doCnt, co1Cnt, howoCnt);
    }
};

template<
    uint32_t mAL1_ = 1,
    uint32_t Kd_ = 1,
    uint32_t Ci1_ = 1
>
struct ConvFmapL1Shape {
    static uint32_t constexpr mAL1 = mAL1_;
    static uint32_t constexpr Kd = Kd_;
    static uint32_t constexpr Ci1 = Ci1_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord()
    {
        return MakeCoord(mAL1, Kd, Ci1);
    }
};

template<
    uint32_t Kd_ = 1,
    uint32_t Ci1_ = 1,
    uint32_t nBL1_ = 1
>
struct ConvFilterL1Shape {
    static uint32_t constexpr Kd = Kd_;
    static uint32_t constexpr Ci1 = Ci1_;
    static uint32_t constexpr nBL1 = nBL1_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord()
    {
        return MakeCoord(Kd, Ci1, nBL1);
    }
};

template <
    uint32_t mL0_ = 1,
    uint32_t kL0_ = 1,
    uint32_t nL0_ = 1
>
struct ConvL0Shape {
    static uint32_t constexpr mL0 = mL0_;
    static uint32_t constexpr kL0 = kL0_;
    static uint32_t constexpr nL0 = nL0_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord()
    {
        return MakeCoord(mL0, kL0, nL0);
    }
};

struct Conv3d6HdCoord : public Coord<4, uint32_t> {
    using Index = uint32_t;

    using Base = Coord<4, Index>;

    static constexpr int N_INDEX = 0;
    static constexpr int D_INDEX = 1;
    static constexpr int C1_INDEX = 2;
    static constexpr int HW_INDEX = 3;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv3d6HdCoord() {}

    CATLASS_HOST_DEVICE
    Conv3d6HdCoord(Coord<4, Index> const &coord) : Base(coord) {}

    CATLASS_HOST_DEVICE
    Conv3d6HdCoord(Index n, Index d, Index c1, Index hw)
        : Base(MakeCoord(n, d, c1, hw)) {}

    CATLASS_HOST_DEVICE
    Index const& n() const
    {
        return this->At(N_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& n()
    {
        return this->At(N_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const& d() const
    {
        return this->At(D_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& d()
    {
        return this->At(D_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const& c1() const
    {
        return this->At(C1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& c1()
    {
        return this->At(C1_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const& hw() const
    {
        return this->At(HW_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& hw()
    {
        return this->At(HW_INDEX);
    }
};

struct Conv3dFracZ3dCoord : public Coord<2, uint32_t> {
    using Index = uint32_t;

    using Base = Coord<2, Index>;

    static constexpr int KDC1KHKW_INDEX = 0;
    static constexpr int N1_INDEX = 1;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv3dFracZ3dCoord() {}

    CATLASS_HOST_DEVICE
    Conv3dFracZ3dCoord(Index kdc1khkw, Index n1)
        : Base(MakeCoord(kdc1khkw, n1)) {}

    CATLASS_HOST_DEVICE
    Index const& kdc1khkw() const
    {
        return this->At(KDC1KHKW_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& kdc1khkw()
    {
        return this->At(KDC1KHKW_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const& n1() const
    {
        return this->At(N1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& n1()
    {
        return this->At(N1_INDEX);
    }
};

/////////////////// Shapes and Coords for Conv2d ///////////////////

template <
    uint32_t Ho_ = 1,
    uint32_t Wo_ = 1,
    uint32_t Cin1_ = 1
>
struct Conv2dFmapL1Shape { // (Ho, Wo, Cin1)
    static constexpr uint32_t Ho = Ho_;
    static constexpr uint32_t Wo = Wo_;
    static constexpr uint32_t Cin1 = Cin1_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord() {
        return MakeCoord(Ho, Wo, Cin1);
    }
};

template <
    uint32_t Cout_ = 16,
    uint32_t Cin1_ = 1
>
struct Conv2dFilterL1Shape { // (Cout, Cin1)
    static constexpr uint32_t Cout = Cout_;
    static constexpr uint32_t Cin1 = Cin1_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<2> ToCoord() {
        return MakeCoord(Cout, Cin1);
    }
};

template <
    uint32_t M_ = 16,
    uint32_t N_ = 16,
    uint32_t K_ = 16
>
struct Conv2dL0Shape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord() {
        return MakeCoord(M, N, K);
    }
};

struct Conv2dFmapCoord : public Coord<5, uint32_t> { // (Batch, C1, H, W, C0)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=5
    using Base = Coord<5, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t BATCH_INDEX = 0;
    static constexpr uint32_t C1_INDEX = 1;
    static constexpr uint32_t H_INDEX = 2;
    static constexpr uint32_t W_INDEX = 3;
    static constexpr uint32_t C0_INDEX = 4;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv2dFmapCoord() {}

    /// Constructs from Coord<5>
    CATLASS_HOST_DEVICE
    Conv2dFmapCoord(Coord<5, Index> const &coord) : Base(coord) {}

    /// Helper to construct from C1, H, W, C0
    CATLASS_HOST_DEVICE
    Conv2dFmapCoord(Index batch, Index c1, Index h, Index w, Index c0)
    : Base(MakeCoord(batch, c1, h, w, c0)) {}

    CATLASS_HOST_DEVICE
    Conv2dFmapCoord(LongIndex batch, LongIndex c1, LongIndex h, LongIndex w, LongIndex c0)
    : Base(MakeCoord(Index(batch), Index(c1), Index(h), Index(w), Index(c0))) {}

    CATLASS_HOST_DEVICE
    Index const &c1() const {
        return this->At(C1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &c1() {
        return this->At(C1_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &batch() const {
        return this->At(BATCH_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &batch() {
        return this->At(BATCH_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &h() const {
        return this->At(H_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &h() {
        return this->At(H_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &w() const {
        return this->At(W_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &w() {
        return this->At(W_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &c0() const {
        return this->At(C0_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &c0() {
        return this->At(C0_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    Conv2dFmapCoord operator+(Base const &b) const {
        return Conv2dFmapCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    Conv2dFmapCoord &operator+=(Base const &b) {
        Base::operator+=(b);
        return *this;
    }
};

struct Conv2dFilterCoord : public Coord<5, uint32_t> { // (Cin1, Kh, Kw, Cout, C0)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=5
    using Base = Coord<5, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t CIN1_INDEX = 0;
    static constexpr uint32_t KH_INDEX = 1;
    static constexpr uint32_t KW_INDEX = 2;
    static constexpr uint32_t COUT_INDEX = 3;
    static constexpr uint32_t C0_INDEX = 4;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv2dFilterCoord() {}

    /// Constructs from Coord<5>
    CATLASS_HOST_DEVICE
    Conv2dFilterCoord(Coord<5, Index> const &coord) : Base(coord) {}

    /// Helper to construct from Cin1, Kh, Kw, Cout, C0
    CATLASS_HOST_DEVICE
    Conv2dFilterCoord(Index cin1, Index kh, Index kw, Index cout, Index c0)
    : Base(MakeCoord(cin1, kh, kw, cout, c0)) {}

    CATLASS_HOST_DEVICE
    Conv2dFilterCoord(LongIndex cin1, LongIndex kh, LongIndex kw, LongIndex cout, LongIndex c0)
    : Base(MakeCoord(Index(cin1), Index(kh), Index(kw), Index(cout), Index(c0))) {}

    CATLASS_HOST_DEVICE
    Index const &cin1() const {
        return this->At(CIN1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cin1() {
        return this->At(CIN1_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &kh() const {
        return this->At(KH_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &kh() {
        return this->At(KH_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &kw() const {
        return this->At(KW_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &kw() {
        return this->At(KW_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &cout() const {
        return this->At(COUT_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cout() {
        return this->At(COUT_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &c0() const {
        return this->At(C0_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &c0() {
        return this->At(C0_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    Conv2dFilterCoord operator+(Base const &b) const {
        return Conv2dFilterCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    Conv2dFilterCoord &operator+=(Base const &b) {
        Base::operator+=(b);
        return *this;
    }
};

struct Conv2dHoWoCoCoord : public Coord<3, uint32_t> { // (Ho, Wo, Cout)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=3
    using Base = Coord<3, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t HO_INDEX = 0;
    static constexpr uint32_t WO_INDEX = 1;
    static constexpr uint32_t COUT_INDEX = 2;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv2dHoWoCoCoord() {}

    /// Constructs from Coord<3>
    CATLASS_HOST_DEVICE
    Conv2dHoWoCoCoord(Coord<3, Index> const &coord) : Base(coord) {}

    /// Helper to construct from Ho, Wo, Cout
    CATLASS_HOST_DEVICE
    Conv2dHoWoCoCoord(Index ho, Index wo, Index cout)
    : Base(MakeCoord(ho, wo, cout)) {}

    CATLASS_HOST_DEVICE
    Conv2dHoWoCoCoord(LongIndex ho, LongIndex wo, LongIndex cout)
    : Base(MakeCoord(Index(ho), Index(wo), Index(cout))) {}

    CATLASS_HOST_DEVICE
    Index const &ho() const {
        return this->At(HO_INDEX);
    }
    
    CATLASS_HOST_DEVICE
    Index &ho() {
        return this->At(HO_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &wo() const {
        return this->At(WO_INDEX);
    }
    
    CATLASS_HOST_DEVICE
    Index &wo() {
        return this->At(WO_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index howo() const {
        return this->At(HO_INDEX) * this->At(WO_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &cout() const {
        return this->At(COUT_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &cout() {
        return this->At(COUT_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    Conv2dHoWoCoCoord operator+(Base const &b) const {
        return Conv2dHoWoCoCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    Conv2dHoWoCoCoord &operator+=(Base const &b) {
        Base::operator+=(b);
        return *this;
    }
};

struct Conv2dCoord : public Coord<5, uint32_t> { // (Batch, H, W, Cout, Cin1)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=5
    using Base = Coord<5, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t BATCH_INDEX = 0;
    static constexpr uint32_t H_INDEX = 1;
    static constexpr uint32_t W_INDEX = 2;
    static constexpr uint32_t COUT_INDEX = 3;
    static constexpr uint32_t CIN1_INDEX = 4;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv2dCoord() {}

    /// Constructs from Coord<5>
    CATLASS_HOST_DEVICE
    Conv2dCoord(Coord<5, Index> const &coord) : Base(coord) {}

    /// Helper to construct from Batch, H, W, Cout, Cin1
    CATLASS_HOST_DEVICE
    Conv2dCoord(Index batch, Index h, Index w, Index cout, Index cin1)
    : Base(MakeCoord(batch, h, w, cout, cin1)) {}

    CATLASS_HOST_DEVICE
    Conv2dCoord(LongIndex batch, LongIndex h, LongIndex w, LongIndex cout, LongIndex cin1)
    : Base(MakeCoord(Index(batch), Index(h), Index(w), Index(cout), Index(cin1))) {}

    CATLASS_HOST_DEVICE
    Index const &batch() const {
        return this->At(BATCH_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &batch() {
        return this->At(BATCH_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &h() const {
        return this->At(H_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &h() {
        return this->At(H_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &w() const {
        return this->At(W_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &w() {
        return this->At(W_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &cout() const {
        return this->At(COUT_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cout() {
        return this->At(COUT_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &cin1() const {
        return this->At(CIN1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cin1() {
        return this->At(CIN1_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    Conv2dCoord operator+(Base const &b) const {
        return Conv2dCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    Conv2dCoord &operator+=(Base const &b) {
        Base::operator+=(b);
        return *this;
    }

    CATLASS_HOST_DEVICE
    auto GetHoWoCoCoord() const {
        return this->GetCoordByAxis<H_INDEX, W_INDEX, COUT_INDEX>();
    }
};

class Conv2dFilterParams {
public:
    typedef uint8_t ShortIndex;
    typedef uint32_t Index;
    static constexpr uint32_t C0 = 16;
    using Ks = Coord<2, ShortIndex>;
    using Pads = Coord<4, ShortIndex>;
    using Strides = Coord<2, ShortIndex>;
    using Dilations = Coord<2, ShortIndex>;
private:
    Ks ks;
    Pads pads;
    Strides strides;
    Dilations dilations;
public:
    Conv2dFilterParams(ShortIndex kh = 0, ShortIndex kw = 0,
        ShortIndex padLeft = 0, ShortIndex padRight = 0, ShortIndex padTop = 0, ShortIndex padBottom = 0,
        ShortIndex strideH = 0, ShortIndex strideW = 0, ShortIndex dilationH = 0, ShortIndex dilationW = 0)
    : ks(MakeCoord(kh, kw)), pads(MakeCoord(padLeft, padRight, padTop, padBottom)), 
      strides(MakeCoord(strideH, strideW)), dilations(MakeCoord(dilationH, dilationW)) {}

    CATLASS_HOST_DEVICE
    ShortIndex const &kh() const {
        return this->ks[0];
    }
    CATLASS_HOST_DEVICE
    ShortIndex &kh() {
        return this->ks[0];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &kw() const {
        return this->ks[1];
    }
    CATLASS_HOST_DEVICE
    ShortIndex &kw() {
        return this->ks[1];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padLeft() const {
        return this->pads[0];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padLeft() {
        return this->pads[0];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padRight() const {
        return this->pads[1];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padRight() {
        return this->pads[1];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padTop() const {
        return this->pads[2];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padTop() {
        return this->pads[2];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padBottom() const {
        return this->pads[3];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padBottom() {
        return this->pads[3];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &strideH() const {
        return this->strides[0];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &strideH() {
        return this->strides[0];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &strideW() const {
        return this->strides[1];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &strideW() {
        return this->strides[1];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &dilationH() const {
        return this->dilations[0];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &dilationH() {
        return this->dilations[0];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &dilationW() const {
        return this->dilations[1];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &dilationW() {
        return this->dilations[1];
    }
};

struct Conv2dParams {
public:
    typedef uint8_t ShortIndex;
    typedef uint32_t Index;
    static constexpr uint32_t C0 = 16;
    using Pads = Coord<4, ShortIndex>;
    using Strides = Coord<2, ShortIndex>;
    using Dilations = Coord<2, ShortIndex>;
private:
    // Batch, Hi, Wi, Cin, Cout, Kh, Kw
    Conv2dFmapCoord fmapShape; // {Batch, Cin1, Hi, Wi, C0}
    Conv2dFilterCoord filterShape; // {Cin1, Kh, Kw, Cout, C0}
    Conv2dFmapCoord outputShape; // {Batch, Cout1, Ho, Wo, C0}
    Conv2dFilterParams configs; // {Ks, Pads, Strides, Dilations}
    Conv2dCoord postIm2colShape; // {Batch, Ho, Wo, Cout, Cin1}
public:
    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv2dParams() {}

    CATLASS_HOST_DEVICE
    Conv2dParams(Index batch, Index hi, Index wi, Index cin, Index cout,
        ShortIndex kh, ShortIndex kw,
        ShortIndex padLeft, ShortIndex padRight, ShortIndex padTop, ShortIndex padBottom,
        ShortIndex strideH, ShortIndex strideW,
        ShortIndex dilationH, ShortIndex dilationW)
    : fmapShape(MakeCoord(batch, CeilDiv(cin, C0), hi, wi, C0)),
      filterShape(MakeCoord(CeilDiv(cin, C0), (Index)kh, (Index)kw, cout, C0)),
      configs(kh, kw, padLeft, padRight, padTop, padBottom, strideH, strideW, dilationH, dilationW)
    {
        Index cout1 = CeilDiv(cout, C0);
        Index ho = (hi + padTop + padBottom - dilationH * (kh - 1) - 1) / strideH + 1;
        Index wo = (wi + padLeft + padRight - dilationW * (kw - 1) - 1) / strideW + 1;
        outputShape = MakeCoord(batch, cout1, ho, wo, C0);
        postIm2colShape = MakeCoord(batch, ho, wo, cout, filterShape.cin1());
    }

    CATLASS_HOST_DEVICE
    static Conv2dParams MakeConv2dParams(
        const Index* dataSizes, // [Batch, Hi, Wi, Cin, Cout]
        const ShortIndex* filterSizes, // [Kh, Kw]
        const ShortIndex* pads, // [padLeft, padRight, padTop, padBottom]
        const ShortIndex* strides, // [strideH, strideW]
        const ShortIndex* dilations) // [dilationH, dilationW]
    {
        return Conv2dParams(
            dataSizes[0], dataSizes[1], dataSizes[2], dataSizes[3], dataSizes[4],
            filterSizes[0], filterSizes[1],
            pads[0], pads[1], pads[2], pads[3],
            strides[0], strides[1], strides[2], strides[3]);
    }

    CATLASS_HOST_DEVICE
    Conv2dFilterParams const &getFilterParams() const {
        return this->configs;
    }

    CATLASS_HOST_DEVICE
    Conv2dFmapCoord const &getOutputShape() const {
        return this->outputShape;
    }

    CATLASS_HOST_DEVICE
    Conv2dCoord const &getPostIm2colShape() const {
        return this->postIm2colShape;
    }

    CATLASS_HOST_DEVICE
    Index const &batch() const {
        return this->fmapShape.batch();
    }
    CATLASS_HOST_DEVICE
    Index &batch() {
        return this->fmapShape.batch();
    }

    CATLASS_HOST_DEVICE
    Index const &hi() const {
        return this->fmapShape.h();
    }
    CATLASS_HOST_DEVICE
    Index &hi() {
        return this->fmapShape.h();
    }

    CATLASS_HOST_DEVICE
    Index const &wi() const {
        return this->fmapShape.w();
    }
    CATLASS_HOST_DEVICE
    Index &wi() {
        return this->fmapShape.w();
    }

    CATLASS_HOST_DEVICE
    Index cin() const {
        return this->fmapShape.c1() * C0;
    }

    CATLASS_HOST_DEVICE
    Index const &cin1() const {
        return this->fmapShape.c1();
    }
    CATLASS_HOST_DEVICE
    Index &cin1() {
        return this->fmapShape.c1();
    }

    CATLASS_HOST_DEVICE
    Index const &cout() const {
        return this->filterShape.cout();
    }
    CATLASS_HOST_DEVICE
    Index &cout() {
        return this->filterShape.cout();
    }

    CATLASS_HOST_DEVICE
    Index const &cout1() const {
        return this->outputShape.c1();
    }
    CATLASS_HOST_DEVICE
    Index &cout1() {
        return this->outputShape.c1();
    }

    CATLASS_HOST_DEVICE
    Index coutRound() const {
        return (this->filterShape.cout() + C0 - 1) / C0 * C0;
    }

    CATLASS_HOST_DEVICE
    Index const &ho() const {
        return this->outputShape.h();
    }
    CATLASS_HOST_DEVICE
    Index &ho() {
        return this->outputShape.h();
    }

    CATLASS_HOST_DEVICE
    Index const &wo() const {
        return this->outputShape.w();
    }
    CATLASS_HOST_DEVICE
    Index &wo() {
        return this->outputShape.w();
    }

    CATLASS_HOST_DEVICE
    Index howo() const {
        return this->outputShape.h() * this->outputShape.w();
    }

    CATLASS_HOST_DEVICE
    Index howoRound() const {
        return (this->howo() + C0 - 1) / C0 * C0;
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &kh() const {
        return this->configs.kh();
    }
    CATLASS_HOST_DEVICE
    ShortIndex &kh() {
        return this->configs.kh();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &kw() const {
        return this->configs.kw();
    }
    CATLASS_HOST_DEVICE
    ShortIndex &kw() {
        return this->configs.kw();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padTop() const {
        return this->configs.padTop();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padTop() {
        return this->configs.padTop();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padBottom() const {
        return this->configs.padBottom();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padBottom() {
        return this->configs.padBottom();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padLeft() const {
        return this->configs.padLeft();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padLeft() {
        return this->configs.padLeft();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padRight() const {
        return this->configs.padRight();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padRight() {
        return this->configs.padRight();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &strideH() const {
        return this->configs.strideH();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &strideH() {
        return this->configs.strideH();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &strideW() const {
        return this->configs.strideW();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &strideW() {
        return this->configs.strideW();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &dilationH() const {
        return this->configs.dilationH();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &dilationH() {
        return this->configs.dilationH();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &dilationW() const {
        return this->configs.dilationW();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &dilationW() {
        return this->configs.dilationW();
    }
};

}

#endif  // CATLASS_CONV_COORD_HPP