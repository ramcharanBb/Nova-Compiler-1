#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"
#include "Compiler/Dialect/nova/NovaOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace nova {

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
using namespace mlir;
inline bool isScalar(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  return !type || type.getRank() == 0;
}

inline SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

Value createIntToFloatCast(Value input, Type targetType, OpBuilder &builder,
                           Location loc) {
  Type inputType = input.getType();
  if (auto inputTensorType = dyn_cast<RankedTensorType>(inputType)) {
    if (isa<IntegerType>(inputTensorType.getElementType())) {
      Value out = builder.create<tensor::EmptyOp>(
          loc, inputTensorType.getShape(), targetType);
      int64_t rank = inputTensorType.getRank();
      AffineMap idMap =
          AffineMap::getMultiDimIdentityMap(rank, builder.getContext());
      SmallVector<AffineMap> maps = {idMap, idMap};
      auto linalgOp = builder.create<linalg::GenericOp>(
          loc, out.getType(), ValueRange{input}, ValueRange{out},
          maps, getNParallelLoopsAttrs(rank),
          [&](OpBuilder &b, Location innerLoc, ValueRange args) {
            Value innerInputScalar = args[0];
            Value inner = b.create<arith::UIToFPOp>(innerLoc, targetType,
                                                    innerInputScalar);
            b.create<linalg::YieldOp>(innerLoc, inner);
          });
      return linalgOp.getResult(0);
    }
  }
  return input;
}

struct NovaOpLinalOp {
  template <typename OpTy>
  static Value mapop(OpTy op, Type resultType, ValueRange args, Value init,
                     OpBuilder &builder) {
    return mapOpImpl(op, resultType, args, init, builder);
  }

private:
  template <typename OpTy>
  static Value mapOpImpl(OpTy op, Type resultType, ValueRange args, Value init,
                         OpBuilder &builder) {
    return nullptr;
  }

  static Value mapOpImpl(nova::AddOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    auto linalgop = builder.create<linalg::AddOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0], args[1]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
    static Value mapOpImpl(nova::SubOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    auto linalgop = builder.create<linalg::SubOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0], args[1]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
    static Value mapOpImpl(nova::MulOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    auto linalgop = builder.create<linalg::MulOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0], args[1]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
      static Value mapOpImpl(nova::DivOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    auto linalgop = builder.create<linalg::DivOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0], args[1]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
        static Value mapOpImpl(nova::PowOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    Location loc = op.getLoc();
    RankedTensorType resultTensorType = dyn_cast<RankedTensorType>(resultType);
    
    if (!resultTensorType) {
        op.emitOpError("PowOp expects ranked tensor result type.");
        return nullptr;
    }

    Type targetFloatType = builder.getF32Type();
    Value lhs = args[0];
    Value rhs = args[1];

    if(isa<IntegerType>(lhs.getType())){
        lhs = createIntToFloatCast(lhs, targetFloatType, builder, loc);
    }
    if(isa<IntegerType>(rhs.getType())){
        rhs = createIntToFloatCast(rhs, targetFloatType, builder, loc);
    }

    auto linalgop = builder.create<linalg::PowFOp>(
        loc, 
        resultType,
        ValueRange{lhs, rhs},
        ValueRange{init}
    );

    return linalgop.getResult(0);
}
  static Value mapOpImpl(nova::SquareOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    auto linalgop = builder.create<linalg::SquareOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
    static Value mapOpImpl(nova::SqrtOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    auto linalgop = builder.create<linalg::SqrtOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
    static Value mapOpImpl(nova::ReciprocalOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    auto linalgop = builder.create<linalg::ReciprocalOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
      static Value mapOpImpl(nova::ExpOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    auto linalgop = builder.create<linalg::ExpOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
    static Value mapOpImpl(nova::LogOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    auto linalgop = builder.create<linalg::LogOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
    
      static Value mapOpImpl(nova::TanhOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    auto linalgop = builder.create<linalg::TanhOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
    static Value mapOpImpl(nova::AbsOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    if(!isa<FloatType>(args[0].getType())){ 
      op->emitOpError("Abs only accepts float or complex types for linalg::AbsOp");
      return nullptr;
    }
    auto linalgop = builder.create<linalg::AbsOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
    static Value mapOpImpl(nova::NegOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
    if(!isa<FloatType>(args[0].getType())){
      op->emitOpError("Neg only accepts float types for linalg::NegFOp");
      return nullptr;
    }
    auto linalgop = builder.create<linalg::NegFOp>(
        op.getLoc(), 
        resultType,
        ValueRange{args[0]},
        ValueRange{init});

    return linalgop.getResult(0);
  }
      static Value mapOpImpl(nova::Rndm2DOp op, Type resultType, ValueRange args,
                         Value init, OpBuilder &builder) {
auto loc=op.getLoc();
//convert args[0] and args[1] to f32

//create a new value by finding a half in args[0] and args[1] .it has to be i32.
    int32_t myFixedSeed = 454496; 
    Value seedVal = builder.create<arith::ConstantOp>(
        loc, 
        builder.getI32Type(), 
        builder.getI32IntegerAttr(myFixedSeed)
    );
auto linalgop = builder.create<linalg::FillRng2DOp>(
    loc,
    ValueRange{args[0],args[1],seedVal},
    ValueRange{init}
);

    return linalgop.getResult(0);
  }
};

template <typename NovaOpTy>
class NovaToLinalgNamedopsConverter : public OpConversionPattern<NovaOpTy> {
public:
  using OpConversionPattern<NovaOpTy>::OpConversionPattern;
  using OpAdaptor = typename NovaOpTy::Adaptor;

  LogicalResult matchAndRewrite(
      NovaOpTy op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto operands = adaptor.getOperands();
    auto resultType = op.getResult().getType();
    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    
    if (!tensorType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor type");

    Value init = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), tensorType.getShape(), tensorType.getElementType());

    Value result = NovaOpLinalOp::mapop(op, resultType, operands, init, rewriter);

    if (!result)
      return rewriter.notifyMatchFailure(op, "mapping failed");

    rewriter.replaceOp(op, result);
    return success();
  }
};
// //RANDOM lowering -Stablehlo reference
// SmallVector<utils::IteratorType, 3> getParallelAndReductionIterators(
//     unsigned nLoops, unsigned nReduction) {
//   SmallVector<utils::IteratorType, 3> res(nLoops - nReduction,
//                                           utils::IteratorType::parallel);
//   res.append(nReduction, utils::IteratorType::reduction);
//   return res;
// }
// Value getEmptySparseTensor(OpBuilder& b, Location loc, ShapedType type,
//                            ArrayRef<Value> dynSizes) {
//   return bufferization::AllocTensorOp::create(
//       b, loc, llvm::cast<TensorType>(type), dynSizes,
//       /*copy=*/Value(),
//       /*memory_space=*/IntegerAttr());
// }

// Value getEmptyTensor(OpBuilder& b, Location loc, ShapedType type,
//                      ArrayRef<Value> dynSizes) {
//   return tensor::EmptyOp::create(
//       b, loc, type.getShape(), type.getElementType(), dynSizes,
//       llvm::cast<RankedTensorType>(type).getEncoding());
// }
// Value getEmptyTensorFor(OpBuilder& b, Location loc, ShapedType resultType,
//                         Operation* op, ValueRange operands) {
//   bool isSparse = sparse_tensor::getSparseTensorEncoding(resultType) != nullptr;
//   // Collect the sizes for a ranked tensor to be passed as parameter to a
//   // new tensor initialization operation. This operation only needs the
//   // dynamic sizes.
//   SmallVector<Value> sizes;
//   if (!resultType.hasStaticShape()) {
//     // Ask the op for its output shape.
//     auto shapeSource = cast<InferShapedTypeOpInterface>(op);
//     SmallVector<Value, 1> reifiedShapes;
//     if (failed(shapeSource.reifyReturnTypeShapes(b, operands, reifiedShapes))) {
//       llvm::report_fatal_error("could not reify");
//     }
//     assert(reifiedShapes.size() == 1 && "Expected one reified result");
//     // Construct sizes for the required dimensions.
//     for (const auto& en : llvm::enumerate(resultType.getShape())) {
//       if (en.value() != ShapedType::kDynamic) continue;
//       sizes.push_back(tensor::ExtractOp::create(
//           b, loc, reifiedShapes[0],
//           ValueRange{b.create<arith::ConstantIndexOp>(loc, en.index())}));
//     }
//   }
//   return isSparse ? getEmptySparseTensor(b, loc, resultType, sizes)
//                   : getEmptyTensor(b, loc, resultType, sizes);
// }

// /// Returns an attribute list that excludes pre-defined attributes.
// template <typename OpTy>
// SmallVector<NamedAttribute> getPrunedAttributeList(OpTy op) {
//   auto elidedAttrs = llvm::to_vector(op.getAttributeNames());
//   if (isa<linalg::LinalgOp>(op.getOperation()))
//     elidedAttrs.push_back(linalg::LinalgDialect::kMemoizedIndexingMapsAttrName);
//   return getPrunedAttributeList(op, elidedAttrs);
// }
// struct RngUniformConversion final
//     : OpConversionPattern<mlir::nova::Rndm2DOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       mlir::nova::Rndm2DOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     // We only handle uniform distributions.
    
//     // TODO(raikonenfnu): Handle other element types as well.
//     auto minTy = dyn_cast<ShapedType>(adaptor.getLhs() .getType());
//     auto maxTy = dyn_cast<ShapedType>(adaptor.getRhs().getType());
//     if (!isa<FloatType>(minTy.getElementType()) ||
//         !isa<FloatType>(maxTy.getElementType())) {
//       return rewriter.notifyMatchFailure(
//           op, "expected min/max for rng op to be FloatType");
//     }
//     auto targetTy = dyn_cast_or_null<ShapedType>(
//         getTypeConverter()->convertType(op.getType()));
//     if (!targetTy) {
//       return rewriter.notifyMatchFailure(
//           op, "expected target shape of rng op to be ShapedType");
//     }
//     auto loc = op.getLoc();
//     Value emptyTensor =
//         getEmptyTensorFor(rewriter, loc, targetTy, op, adaptor.getOperands());
//     // Creates index map using target matrix's rank.
//     auto targetRank = targetTy.getRank();
//     SmallVector<AffineMap, 3> indexingMaps(
//         2, AffineMap::get(targetRank, /*symbolCount=*/0,
//                           SmallVector<AffineExpr>({}), rewriter.getContext()));
//     indexingMaps.push_back(rewriter.getMultiDimIdentityMap(targetRank));
//     const int kInitialSeed = 0;

//     // Generic region with LCG Algorithm that make use of element index from:
//     // https://reviews.llvm.org/D101364
//     auto linalgOp = rewriter.create<linalg::GenericOp>(
//         loc, /*resultTensors=*/targetTy,
//         /*inputs=*/
//         ValueRange{adaptor.getOperands()[0], adaptor.getOperands()[1]},
//         /*outputs=*/emptyTensor, indexingMaps,
//         getParallelAndReductionIterators(/*nLoops=*/targetRank,
//                                          /*nReduction=*/0),
//         [&](OpBuilder &b, Location loc, ValueRange args) {
//           llvm::SmallVector<Value> updateVec = {b.create<arith::ConstantOp>(
//               loc, b.getI32IntegerAttr(kInitialSeed))};
//           Value multiplier =
//               b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(1103515245));
//           Value incrementStep =
//               b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(12345));
//           // For output matrix with rank N:
//           // temp1 = (cast(I32, index(D.0)) + seed) * mult + incr
//           // ...
//           // tempN = (cast(I32, index(D.(N))) + tempN_1) * mult + incr
//           for (int i = 0; i < targetRank; i++) {
//             Value update = updateVec.back();
//             Value ind = b.create<linalg::IndexOp>(loc, i);
//             Value castInd =
//                 b.create<arith::IndexCastOp>(loc, b.getI32Type(), ind);
//             Value addRes = b.create<arith::AddIOp>(loc, castInd, update);
//             Value multRes = b.create<arith::MulIOp>(loc, addRes, multiplier);
//             Value incRes = b.create<arith::AddIOp>(loc, multRes, incrementStep);
//             updateVec.push_back(incRes);
//           }
//           // Scaling = (max - min) * const(F64, 2.3283064E-10)
//           // which is derived from rand(min,max) = rand()/(RAND_MAX/(max-min)).
//           Value epsilon = b.create<arith::ConstantOp>(
//               loc, b.getFloatAttr(args[0].getType(), 2.3283064E-10));
//           Value range = b.create<arith::SubFOp>(loc, args[1], args[0]);
//           Value scale = b.create<arith::MulFOp>(loc, range, epsilon);
//           // Res = cast(T, cast(F64, tempN) * scaling + min)
//           Value updateCast = b.create<arith::UIToFPOp>(
//               loc, targetTy.getElementType(), updateVec.back());
//           Value scaleUpdate = b.create<arith::MulFOp>(loc, updateCast, scale);
//           Value res = b.create<arith::AddFOp>(loc, scaleUpdate, args[0]);
//           b.create<linalg::YieldOp>(loc, res);
//         },
//         getPrunedAttributeList(op));
//     rewriter.replaceOp(op, linalgOp.getResults());
//     return success();
//   }
// };
void populateNovaToLinalgNamedPatterns(RewritePatternSet &patterns) {
  patterns.add<
           //NovaToLinalgNamedopsConverter<nova::AddOp>,
          //  NovaToLinalgNamedopsConverter<nova::SubOp>,
            NovaToLinalgNamedopsConverter<nova::MulOp>,
             NovaToLinalgNamedopsConverter<nova::DivOp>,
            NovaToLinalgNamedopsConverter<nova::SqrtOp>,
            NovaToLinalgNamedopsConverter<nova::NegOp>,
            NovaToLinalgNamedopsConverter<nova::AbsOp>,
             NovaToLinalgNamedopsConverter<nova::ExpOp>,
             NovaToLinalgNamedopsConverter<nova::LogOp>,
             NovaToLinalgNamedopsConverter<nova::TanhOp>,
             NovaToLinalgNamedopsConverter<nova::PowOp>,
             NovaToLinalgNamedopsConverter<nova::MinOp>,
             NovaToLinalgNamedopsConverter<nova::MaxOp>,
             NovaToLinalgNamedopsConverter<nova::ReciprocalOp>,
            NovaToLinalgNamedopsConverter<nova::SquareOp>,
            NovaToLinalgNamedopsConverter<nova::Rndm2DOp>
           // RngUniformConversion

  >(patterns.getContext());
}

} 
}