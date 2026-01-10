#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"

#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Translation/NovaToTosa/NovaToTosa.h"
// base function
namespace mlir {
namespace nova {
struct NovaOpTosaOp {
  static Value rmappingtosa(nova::AddOp op, Type resultType, ValueRange input,
                            OpBuilder *builder) {

    return builder->create<tosa::AddOp>(op.getLoc(), resultType, input[0], input[1]);
  }

  static Value rmappingtosa(nova::SubOp op, Type resultType, ValueRange input,
                            OpBuilder *builder) {

    auto restensor = dyn_cast<mlir::TensorType>(resultType);
    return builder->create<tosa::SubOp>(op.getLoc(), resultType, input[0], input[1]);
  }
  static Value rmappingtosa(nova::MulOp op, Type resultType, ValueRange input,
                            OpBuilder *builder) {

    auto restensor = cast<mlir::RankedTensorType>(resultType);
    Value v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
    Value w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);
    Value init = builder->create<tensor::EmptyOp>(
        op.getLoc(), restensor.getShape(), restensor.getElementType(),
        restensor.getEncoding());

    return builder
        ->create<linalg::MulOp>(op.getLoc(), resultType, ValueRange{v, w},
                                ValueRange{init})
        .getResult(0);
  }
  static Value rmappingtosa(nova::PowOp op, Type resultType, ValueRange input,
                            OpBuilder *builder) {

    auto restensor = dyn_cast<mlir::TensorType>(resultType);
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
    auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);

    return builder->create<tosa::PowOp>(op.getLoc(), resultType, v, w);
  }
  static Value rmappingtosa(nova::SqrtOp op, Type resultType, ValueRange input,
                            OpBuilder *builder) {
    auto restensor = dyn_cast<mlir::TensorType>(resultType);
    auto elemType = restensor.getElementType();
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
    // Create constant 0.5 tensor
    DenseElementsAttr halfAttr;
    if (elemType.isF32()) {
      halfAttr =
          DenseElementsAttr::get(restensor, builder->getF32FloatAttr(0.5f));
    } else {
      op.emitError("Sqrt lowering only supports floating-point tensors");
      return nullptr;
    }

    Value half =
        builder->create<tosa::ConstOp>(op.getLoc(), restensor, halfAttr);
    return builder->create<tosa::PowOp>(op.getLoc(), resultType, v, half);
  }

  static Value rmappingtosa(nova::SquareOp op, Type resultType,
                            ValueRange input, OpBuilder *builder) {
    auto restensor = dyn_cast<mlir::TensorType>(resultType);
    auto elemType = restensor.getElementType();
    auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
    // Create constant 2.0 tensor
    DenseElementsAttr twoAttr;
    if (elemType.isF32()) {
      twoAttr =
          DenseElementsAttr::get(restensor, builder->getF32FloatAttr(2.0f));
    } else {
      op.emitError("Square lowering only supports floating-point tensors");
      return nullptr;
    }

    Value two = builder->create<tosa::ConstOp>(op.getLoc(), restensor, twoAttr);
    return builder->create<tosa::PowOp>(op.getLoc(), resultType, v, two);
  }

  template <typename OpTy>
  static Value rmaptop(OpTy op, Type resultType, ValueRange input,
                       OpBuilder *builder) {
    return rmappingtosa(op, resultType, input, builder);
  }

private:
  template <typename OpTy>
  static Value rmappingtosa(OpTy op, Type resultType, ValueRange input,
                            OpBuilder *builder) {
    return nullptr;
  }
};

// creating a template
template <typename NovaTopTy>
class NovaToTosaLoweringTemplater : public OpConversionPattern<NovaTopTy> {
public:
  using OpConversionPattern<NovaTopTy>::OpConversionPattern;
  using OpAdaptor = typename NovaTopTy::Adaptor; // for getting all meta data
                                                 // dynamically using adaptor
  LogicalResult
  matchAndRewrite(NovaTopTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange operands = adaptor.getOperands();
    // checking operand is empty or not
    if (operands.empty())
      return rewriter.notifyMatchFailure(
          op, "expected operands for tosa lowering operations");
    // getting resultType
    auto resultType = op.getResult().getType();
    Value result = NovaOpTosaOp::rmaptop(op, resultType, operands, &rewriter);
    if (!result)
      return rewriter.notifyMatchFailure(op, "failed to map to TOSA operation");

    rewriter.replaceOp(op, result);
    return success();
  }
};
void populateNovaToTosaTemplatePatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<NovaToTosaLoweringTemplater<nova::AddOp>,
               NovaToTosaLoweringTemplater<nova::SubOp>,
               NovaToTosaLoweringTemplater<nova::MulOp>,
               NovaToTosaLoweringTemplater<nova::SquareOp>,
               NovaToTosaLoweringTemplater<nova::PowOp>,
               NovaToTosaLoweringTemplater<nova::SqrtOp>>(
      patterns.getContext());
}
} // namespace nova
} // namespace mlir