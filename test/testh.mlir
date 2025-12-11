output = rewriter.create<GenericOp>(
            loc,
            newType,
            ValueRange{inputBiasValue},
            ValueRange{initTensor},
            maps,
            iterators,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value castedValue;
              // Assuming you want an extension if the target bitwidth is larger
              if (newElementType.getIntOrFloatBitWidth() > biasType.getElementType().getIntOrFloatBitWidth()) {
                  // Choose signed or unsigned extension as appropriate for your use case
                  // This example uses Signed Integer Extension
                  castedValue = b.create<arith::ExtSIOp>(loc, newElementType, args[0]);
              } else if (newElementType.getIntOrFloatBitWidth() < biasType.getElementType().getIntOrFloatBitWidth()){
                  // Handle truncation if the target bitwidth is smaller
                  castedValue = b.create<arith::TruncIOp>(loc, newElementType, args[0]);
              } else {
                  // Bitwidths are same, perhaps just need a bitcast for signedness or just use the value directly
                  // The simple case: just yield the input if types are identical in width
                  castedValue = args[0]; 
              }
              
              b.create<linalg::YieldOp>(loc, castedValue);
            }
          ).getResult(0); 
        }
      }
      
    Value output = transformedBias;