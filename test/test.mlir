module{
<<<<<<< HEAD
   func.func @main1(%arg0: tensor<256x256xf32>,%arg1:tensor<256x256xf32>,%arg2:tensor<256x256xf32> ) -> tensor<256x256xf32> {
  %1 =nova.matmul %arg0,%arg1 : tensor<256x256xf32>,tensor<256x256xf32>
  %2 = nova.add %1,%arg2 : tensor<256x256xf32>,tensor<256x256xf32>
  return %2 :tensor<256x256xf32>
=======
   func.func @main1(%arg0: tensor<256x256x256xf32>,%arg1:tensor<256x256x256xf32>,%arg2:tensor<256x256x256xf32> ) -> tensor<256x256x256xf32> {
  %1 =nova.matmul %arg0,%arg1 : tensor<256x256x256xf32>,tensor<256x256x256xf32>
  %2 = nova.add %1,%arg2 : tensor<256x256x256xf32>,tensor<256x256x256xf32>
  return %2 :tensor<256x256x256xf32>
>>>>>>> c761c4cfe2910694135a142d37b01ff6aac30e4f
  }
}
  
