#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::nova;

#define GET_OP_CLASSES
#include "Compiler/Dialect/nova/NovaOps.cpp.inc"



//--------------------------------------------------------------
// Helper inferreturn function for binary operations
//---------------------------------------------------------------

/*constraints on input : 
  1 -> 2 operands 
  2 -> They need to be same or compatible type 
  3 -> compatible means they can be made same by broadcasting
  4 -> should supported data types : int,float,

  constraint on output:
  1-> result and operands type should be same
  2 -> if broadcaseted result type needs to be same as operands type after broadcasting*/
template<typename BinOpinfertemplate>
static LogicalResult BinaryInferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
   //create a function specific for this operation and call it here
  // Check we have exactly 2 operands
  if (operands.empty() || operands.size() != 2) {
      return mlir::emitError(*loc) << BinOpinfertemplate::getOperationName() 
                            << " requires exactly 2 operands";}
  // Get the types of the operands
  auto lhsType = llvm::dyn_cast<TensorType>(operands[0].getType());
  auto rhsType = llvm::dyn_cast<TensorType>(operands[1].getType());
  // Verify both operands are tensors
  if (!lhsType || !rhsType) {
    if (loc){
      mlir::emitError(*loc) << BinOpinfertemplate::getOperationName() 
                            << " operands must be tensor types";
      return failure();
    }
    return failure();
  } 
  
  if (lhsType.getElementType()!= rhsType.getElementType()) {
    if (loc) {
      mlir::emitError(*loc) << BinOpinfertemplate::getOperationName() 
                            << " operands must have the same element type";
    }
    return failure();
  }

  // Check if the operand types are the same
  if(operands[0].getType() == operands[1].getType()){
    inferredReturnTypes.push_back(operands[0].getType());
    return success();
  }

  // we assume result type is same as lhs type
  inferredReturnTypes.push_back(operands[0].getType());
  
  return success();
    }

//-------------------------------------------------    
//addOp   
//------------------------------------------------- 
LogicalResult AddOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes<AddOp>(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }

//-------------------------------------------------    
//SubOp  
//------------------------------------------------- 
LogicalResult SubOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes<SubOp>(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }

//-------------------------------------------------    
//MulOp     
//------------------------------------------------- 
LogicalResult MulOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes<MulOp>(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }

//-------------------------------------------------    
//DivideOp     
//------------------------------------------------- 

LogicalResult DivOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes<DivOp>(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }

//-------------------------------------------------    
//remainderOp     
//------------------------------------------------- 
LogicalResult RemOp::inferReturnTypes(     
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes<RemOp>(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }

//-------------------------------------------------    
//powerOp     
//------------------------------------------------- 

LogicalResult PowOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes<PowOp>(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }

//-------------------------------------------------    
//maximumOp     
//-------------------------------------------------    
LogicalResult MaxOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes<MaxOp>(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }

//-------------------------------------------------    
//minimumOp
//-------------------------------------------------
LogicalResult MinOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes<MinOp>(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }
//-------------------------------------------------    
//andOp
//-------------------------------------------------
LogicalResult AndOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes<AndOp>(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }
//-------------------------------------------------    
//orOp
//-------------------------------------------------
LogicalResult OrOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes<OrOp>(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }
//-------------------------------------------------    
//xorOp
//-------------------------------------------------
LogicalResult XorOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
      return BinaryInferReturnTypes<XorOp>(context, loc, operands, attributes, properties, regions, inferredReturnTypes);
    }
