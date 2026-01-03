module {
  memref.global "private" @v : memref<1024xf32> = dense<0.0>
  func.func @main1() -> memref<1024xf32> {
    %0 = memref.get_global @v : memref<1024xf32>
    return %0 : memref<1024xf32>
  }
}
