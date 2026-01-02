; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%0 = type { ptr, ptr, i64, i64, i64, i64, i64, ptr, ptr, i64, i64, i64, i64, i64, ptr, ptr, i64, i64, i64, i64, i64, i64, i64, i64 }

@__constant_4x3xf32_0 = private constant [4 x [3 x float]] [[3 x float] [float 1.000000e+01, float 1.100000e+01, float 1.200000e+01], [3 x float] [float 0x3FB99999A0000000, float 0x3FC99999A0000000, float 0x3FD3333340000000], [3 x float] [float 1.300000e+01, float 1.400000e+01, float 1.500000e+01], [3 x float] [float 1.600000e+01, float 1.700000e+01, float 1.800000e+01]], align 64
@__constant_3x3xf32 = private constant [3 x [3 x float]] [[3 x float] [float 0x3FB99999A0000000, float 0x3FC99999A0000000, float 0x3FD3333340000000], [3 x float] [float 0x3FD99999A0000000, float 5.000000e-01, float 0x3FE3333340000000], [3 x float] [float 0x40119999A0000000, float 5.500000e+00, float 0x401A666660000000]], align 64
@__constant_4x3xf32 = private constant [4 x [3 x float]] [[3 x float] [float 0x3FF19999A0000000, float 0x40019999A0000000, float 0x400A666660000000], [3 x float] [float 0x40119999A0000000, float 5.500000e+00, float 0x401A666660000000], [3 x float] [float 1.300000e+01, float 1.400000e+01, float 1.500000e+01], [3 x float] [float 0x401ECCCCC0000000, float 0x40219999A0000000, float 0x4023CCCCC0000000]], align 64
@main1_kernel_binary = internal constant [5144 x i8] c"P\EDU\BA\01\00\10\00\08\14\00\00\00\00\00\00\02\00\01\01@\00\00\00(\11\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\01\00V\00\00\00\00\00\00\00\00\00\00\00\11\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\7FELF\02\01\01A\08\00\00\00\00\00\00\00\02\00\BE\00\01\00\00\00\00\00\00\00\00\00\00\00\80\10\00\00\00\00\00\00\00\0D\00\00\00\00\00\00\04V\00\06@\008\00\03\00@\00\0E\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.note.nv.tkinfo\00.note.nv.cuinfo\00.nv.info\00.text.main1_kernel\00.nv.info.main1_kernel\00.nv.shared.main1_kernel\00.nv.constant0.main1_kernel\00.rel.nv.constant0.main1_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.note.nv.tkinfo\00.note.nv.cuinfo\00.nv.info\00.text.main1_kernel\00.nv.info.main1_kernel\00.nv.shared.main1_kernel\00.rel.nv.constant0.main1_kernel\00.nv.constant0.main1_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00main1_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\03\00\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\009\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00R\00\00\00\03\00\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B2\00\00\00\03\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\CD\00\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\FD\00\00\00\03\00\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\19\01\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\01\00\00\12\10\0D\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\04\04\00\00\00\04\14\00\00\00\0C\81\80\80(\00\04\A4\00\00\00\00\00\00\00\00\00\00\0C\00\00\00\8C\00\00\00\D0\07\00\00NVIDIA Corp\00\02\00\00\00\00\00\00\00\01\00\00\00\07\00\00\006\00\00\00`\00\00\00\00ptxas\00Cuda compilation tools, release 13.0, V13.0.88\00Build cuda_13.0.r13.0/compiler.36424714_0\00-O 2 -arch sm_86 \00\00\00\0C\00\00\00\08\00\00\00\E8\03\00\00NVIDIA Corp\00\02\00V\00\82\00\00\00\04/\08\00\08\00\00\00\14\00\00\00\04\11\08\00\08\00\00\00\00\00\00\00\04\12\08\00\08\00\00\00\00\00\00\00\047\04\00\82\00\00\00\015\00\00\04\0A\08\00\04\00\00\00`\01\C0\00\03\19\C0\00\04\17\0C\00\00\00\00\00\17\00\B8\00\00\F0!\00\04\17\0C\00\00\00\00\00\16\00\B0\00\00\F0!\00\04\17\0C\00\00\00\00\00\15\00\A8\00\00\F0!\00\04\17\0C\00\00\00\00\00\14\00\A0\00\00\F0!\00\04\17\0C\00\00\00\00\00\13\00\98\00\00\F0!\00\04\17\0C\00\00\00\00\00\12\00\90\00\00\F0!\00\04\17\0C\00\00\00\00\00\11\00\88\00\00\F0!\00\04\17\0C\00\00\00\00\00\10\00\80\00\00\F0!\00\04\17\0C\00\00\00\00\00\0F\00x\00\00\F5!\00\04\17\0C\00\00\00\00\00\0E\00p\00\00\F5!\00\04\17\0C\00\00\00\00\00\0D\00h\00\00\F0!\00\04\17\0C\00\00\00\00\00\0C\00`\00\00\F0!\00\04\17\0C\00\00\00\00\00\0B\00X\00\00\F0!\00\04\17\0C\00\00\00\00\00\0A\00P\00\00\F0!\00\04\17\0C\00\00\00\00\00\09\00H\00\00\F0!\00\04\17\0C\00\00\00\00\00\08\00@\00\00\F5!\00\04\17\0C\00\00\00\00\00\07\008\00\00\F5!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F0!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F0!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F5!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F5!\00\03\1B\FF\00\03_\00\00\04\1C\08\00P\00\00\00\F0\02\00\00\04\05\0C\00\01\00\00\00\01\00\00\00\01\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FFs\00\00\00\00\00\00\00\00\00\00\11%\00\056D\00\00\00\00\00\00\00\02\00\00\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$v\01\FF\00\0A\00\00\FF\00\8E\07\00\E4\0F\00$v\0D\FF\00\82\00\00\FF\00\8E\07\00\E4\0F\00$v\00\FF\00\83\00\00\FF\00\8E\07\00\C6\0F\00\0Cz\00\0D\00\84\00\00p`\F0\03\00\C8\0F\00\0Cz\00\00\00\85\00\00\00c\F0\03\00\DA\0F\00M\09\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\19y\02\00\00\00\00\00\00&\00\00\00\22\0E\00\B9z\04\00\00F\00\00\00\0A\00\00\00\C6\0F\00\19y\09\00\00\00\00\00\00%\00\00\00b\0E\00\19x\06\02\02\00\00\00\FF\06\00\00\00\E4\1F\00\19x\07\FF\1E\00\00\00\02\16\01\00\00\E4\0F\00\10z\02\06\00v\00\00\FF\E0\F1\07\00\E2\0F\00$x\05\09\03\00\00\00\FF\02\8E\07\00\C6/\00\10z\03\07\00w\00\00\FF\E4\7F\00\00\CA\0F\00%x\02\05\04\00\00\00\02\00\8E\07\00\CA\0F\00\81y\0B\02\04\00\00\00\00\19\1E\0C\00b\01\00\19x\05\0D\02\00\00\00\00\02\01\00\00\E2\0F\04$x\04\0D\04\00\00\00\FF\00\8E\07\00\E2\0F\04\02x\0A\00\02\00\00\00\00\0F\00\00\00\E2\0F\00%x\06\0D\0C\00\00\00\06\00\8E\07\00\C8\0F\00%x\04\09\0C\00\00\00\04\00\8E\07\00\E2\0F\00\10z\06\06\00h\00\00\FF\E0\F3\07\00\C6\0F\00$x\09\00\0C\00\00\00\FF\02\8E\07\00\E2\0F\00\10z\04\04\00Z\00\00\FF\E0\F1\07\00\E2\0F\00$v\11\FF\00\86\00\00\FF\00\8E\07\00\E4\0F\00$v\0C\FF\00\82\00\00\FF\00\8E\07\00\E2\0F\00\10z\05\05\00[\00\00\FF\E4\7F\00\00\E2\0F\00$v\0D\FF\00\83\00\00\FF\00\8E\07\00\E2\0F\00\10z\07\07\00i\00\00\09\E4\FF\00\00\E4\0F\00\19v\0A\11\00\87\00\00\0A\02\01\00\00\E4\1F\00\81y\00\04\04\00\00\00\00\19\1E\0C\00\A8\10\00\81y\09\06\04\00\00\00\00\19\1E\0C\00\A2\0E\00\10z\0C\0C\00\86\00\00\FF\E0\F1\07\00\C4\0F\00\02z\0F\00\00\87\00\00\00\0F\00\00\00\E4\0F\00\10z\0D\0D\00\87\00\00\FF\E4\7F\00\00\E4\0F\00\0Cz\00\0C\00\84\00\00p`\F0\03\00\E2\0F\00$x\0F\0F\0C\00\00\00\FF\02\8E\07\00\E2\0F\00\11r\04\11\04\00\00\00\FF\10\82\07\00\E4\1F\00\0Cz\00\0D\00\85\00\00\00c\F0\03\00\E4\0F\00\10r\05\05\0A\00\00\00\FF\E4\FF\00\00\E2\0F\00 r\00\00\09\00\00\00\00\00@\00\00\E2O\00%x\08\11\0C\00\00\00\06\00\8E\07\00\C6\0F\00!r\0B\00\0B\00\00\00\00\00\00\00\00\E2\0F\02$x\07\09\01\00\00\00\0F\02\8E\07\00\E4\0F\00$r\06\FF\FF\00\00\00\08\00\8E\07\00\E4\0F\00\86y\00\02\0B\00\00\00\04\19\10\0C\00\E2\01\00G\89\00\00\F0\FE\FF\FF\FF\FF\83\03\00\EA\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00(\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00h\01\00\00\00\00\00\005\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A0\02\00\00\00\00\00\00\D8\00\00\00\00\00\00\00\02\00\00\00\08\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\CD\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\03\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\07\00\00\00\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\00\A4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\009\00\00\00\07\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\8C\04\00\00\00\00\00\00 \00\00\00\00\00\00\00\05\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00I\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\AC\04\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00e\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\04\00\00\00\00\00\00\C0\01\00\00\00\00\00\00\03\00\00\00\0D\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\FD\00\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\90\06\00\00\00\00\00\00 \00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\19\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B0\06\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\DA\00\00\00\09\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C0\06\00\00\00\00\00\00\10\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\93\00\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\06\00\00\00\00\00\00 \02\00\00\00\00\00\00\00\00\00\00\0D\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00R\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\09\00\00\00\00\00\00\00\04\00\00\00\00\00\00\03\00\00\00\08\00\00\14\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\80\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\D0\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\000\06\00\00\00\00\00\000\06\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\80\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\01\01P\00\00\00P\02\00\00\00\00\00\00K\02\00\00@\00\00\00\01\00\07\00V\00\00\00\00\00\00\00\00\00\00\00\11\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C7\08\00\00\00\00\00\00H\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\B5/\FD`\C7\07\0D\12\00V\D8J\22@M\F4\DC\92\9B\A02\F4\03\E5\0F\A9h\9A\F1\09{1CVyV\EE\FC\F1\C4\05\D5*\EE\E8\D5\01@\00@\00C\00\D2\03\C9\D7<\EB\D4\BC\98Cs\F4\E6\9Bz\05d\94flp\\\82^\C4\12\B3J\B19J\06D\A1\C58N@\D4\A14N\08_/kND\B9\C6g\BE\DA\EB\BD\B0\D4\80\E8\8C\0F\BD\BB\1B\CA5\09\8C,\0E\E5\AE+x\B5\80\F3\E0\C5\DC\17\F3\E0W\0CW\A0\16\D3\C10\CE\A9\F9\89\B3\97b\11g\08U)\E2E\BD\03\AB\DE\93g\DA\C2|\B9zO\9Ci\0BS\96zG\12\C7q\0E\92\81\AE[\13J4\84^U8\18Y\0E\FD\D0\88\D65\EA\9Br\92\83\03\C7A\08\E07\19\EC\D3a\B0nQ\13\ED\F8\F4T#]\97(\FFW\DC?yD>FC>\DD\A8\FD\EA\83\C2\BA\A61\EE\1E\00[\B1\EBj\9CK\A0\FE\ABd\96V\DD\DBBK\93\F9\D6$j!x\A6=\E3\0D\BE%Qla\EFJ\E3\0BQ\16#\EC#\84 \14\C2h7dK\87h]\84\F2\0F'\84o\DD3\B2\02\80\83\A0!ajFAA\92\92B6\0Ep\02\1123\0F \18v@\D2e+\96{&\A1\D1\14\A6\C8~\F9\8A\0C\0A\B9\7FR\1D\83\CA3\A4b\0E\1B\B9\0A\DA\02\F3)\98#d\D3\0C\91\14\C9\DF!\A6\CC\CF\A0\A1\0Dm: /\F9\C2\AB^ld\C3\0E~S\97\B8\F9\18m`\16\AF\13\FB\BB\06m\85]n\B1U\8F\0E\D2o\83\9D\F2\F3\1B\C2\A2\8Fa\AF\D5\15\D2\BC\DB\E5\9C1\16\B7@\A5\0C\8FH\F9\B7\86\C1A\95\F7\FFJ\DF\09\EB\98\E8\81\DA{\C5%\E3\A7\11~\CAf)\A3#\86\9B_\F9\87\1D\F9#\96i\D0&c\04\A6$7\9A\DE\9D\D3tn6\B8ac\E0\82x\96Ry\07}\94\9Fz{S5,\7F]L\B9$\D4\85\99GO\FCU\82\D1\1Eiw\AF^\FE\DAv'\0Et\E68\E79\9A9V\E24\DA\08\C2\89\E0`g\A3c\C9\B8\90Q\E6\A2t\8Fv\C4\F1F\1B/'.\D7\E7\CCI\A1\A9\95S]\0ET\0D\00\00\00\00\00", align 8
@main1_kernel_module = internal global ptr null
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 123, ptr @main1_kernel_load, ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 123, ptr @main1_kernel_unload, ptr null }]
@main1_kernel_main1_kernel_name = private unnamed_addr constant [13 x i8] c"main1_kernel\00", align 1

declare i32 @cudaMalloc(ptr, i64)

declare i32 @cudaMemcpy(ptr, ptr, i64, i32)

declare i32 @cudaFree(ptr)

declare void @dealloc_helper(ptr, ptr, ptr, ptr, ptr)

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @main1() {
  %1 = call ptr @malloc(i64 112)
  %2 = ptrtoint ptr %1 to i64
  %3 = add i64 %2, 63
  %4 = urem i64 %3, 64
  %5 = sub i64 %3, %4
  %6 = inttoptr i64 %5 to ptr
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %1, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, ptr %6, 1
  %9 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, i64 0, 2
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %9, i64 4, 3, 0
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, i64 3, 3, 1
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 3, 4, 0
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 1, 4, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %6, ptr @__constant_4x3xf32_0, i64 48, i1 false)
  %14 = alloca ptr, align 8
  %15 = call i32 @cudaMalloc(ptr %14, i64 48)
  %16 = load ptr, ptr %14, align 8
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %16, 0
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, ptr %16, 1
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 0, 2
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, [2 x i64] [i64 4, i64 3], 3
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, [2 x i64] undef, 4
  %22 = call i32 @cudaMemcpy(ptr %16, ptr @__constant_4x3xf32, i64 48, i32 4)
  %23 = alloca ptr, align 8
  %24 = call i32 @cudaMalloc(ptr %23, i64 36)
  %25 = load ptr, ptr %23, align 8
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %25, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, ptr %25, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 0, 2
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, [2 x i64] [i64 3, i64 3], 3
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, [2 x i64] undef, 4
  %31 = call i32 @cudaMemcpy(ptr %25, ptr @__constant_3x3xf32, i64 36, i32 4)
  %32 = alloca ptr, align 8
  %33 = call i32 @cudaMalloc(ptr %32, i64 48)
  %34 = load ptr, ptr %32, align 8
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %34, 0
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, ptr %34, 1
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 0, 2
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, [2 x i64] [i64 4, i64 3], 3
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, [2 x i64] undef, 4
  %40 = call i32 @cudaMemcpy(ptr %34, ptr %6, i64 48, i32 4)
  %41 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 0
  %42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 1
  %43 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 4, 0
  %44 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, 4, 1
  %45 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 4, 0
  %46 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, 4, 1
  %47 = alloca %0, align 8
  %48 = alloca ptr, i64 24, align 8
  %49 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 0
  store ptr %16, ptr %49, align 8
  %50 = getelementptr ptr, ptr %48, i32 0
  store ptr %49, ptr %50, align 8
  %51 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 1
  store ptr %16, ptr %51, align 8
  %52 = getelementptr ptr, ptr %48, i32 1
  store ptr %51, ptr %52, align 8
  %53 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 2
  store i64 0, ptr %53, align 4
  %54 = getelementptr ptr, ptr %48, i32 2
  store ptr %53, ptr %54, align 8
  %55 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 3
  store i64 4, ptr %55, align 4
  %56 = getelementptr ptr, ptr %48, i32 3
  store ptr %55, ptr %56, align 8
  %57 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 4
  store i64 3, ptr %57, align 4
  %58 = getelementptr ptr, ptr %48, i32 4
  store ptr %57, ptr %58, align 8
  %59 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 5
  store i64 %41, ptr %59, align 4
  %60 = getelementptr ptr, ptr %48, i32 5
  store ptr %59, ptr %60, align 8
  %61 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 6
  store i64 %42, ptr %61, align 4
  %62 = getelementptr ptr, ptr %48, i32 6
  store ptr %61, ptr %62, align 8
  %63 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 7
  store ptr %25, ptr %63, align 8
  %64 = getelementptr ptr, ptr %48, i32 7
  store ptr %63, ptr %64, align 8
  %65 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 8
  store ptr %25, ptr %65, align 8
  %66 = getelementptr ptr, ptr %48, i32 8
  store ptr %65, ptr %66, align 8
  %67 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 9
  store i64 0, ptr %67, align 4
  %68 = getelementptr ptr, ptr %48, i32 9
  store ptr %67, ptr %68, align 8
  %69 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 10
  store i64 3, ptr %69, align 4
  %70 = getelementptr ptr, ptr %48, i32 10
  store ptr %69, ptr %70, align 8
  %71 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 11
  store i64 3, ptr %71, align 4
  %72 = getelementptr ptr, ptr %48, i32 11
  store ptr %71, ptr %72, align 8
  %73 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 12
  store i64 %43, ptr %73, align 4
  %74 = getelementptr ptr, ptr %48, i32 12
  store ptr %73, ptr %74, align 8
  %75 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 13
  store i64 %44, ptr %75, align 4
  %76 = getelementptr ptr, ptr %48, i32 13
  store ptr %75, ptr %76, align 8
  %77 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 14
  store ptr %34, ptr %77, align 8
  %78 = getelementptr ptr, ptr %48, i32 14
  store ptr %77, ptr %78, align 8
  %79 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 15
  store ptr %34, ptr %79, align 8
  %80 = getelementptr ptr, ptr %48, i32 15
  store ptr %79, ptr %80, align 8
  %81 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 16
  store i64 0, ptr %81, align 4
  %82 = getelementptr ptr, ptr %48, i32 16
  store ptr %81, ptr %82, align 8
  %83 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 17
  store i64 4, ptr %83, align 4
  %84 = getelementptr ptr, ptr %48, i32 17
  store ptr %83, ptr %84, align 8
  %85 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 18
  store i64 3, ptr %85, align 4
  %86 = getelementptr ptr, ptr %48, i32 18
  store ptr %85, ptr %86, align 8
  %87 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 19
  store i64 %45, ptr %87, align 4
  %88 = getelementptr ptr, ptr %48, i32 19
  store ptr %87, ptr %88, align 8
  %89 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 20
  store i64 %46, ptr %89, align 4
  %90 = getelementptr ptr, ptr %48, i32 20
  store ptr %89, ptr %90, align 8
  %91 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 21
  store i64 0, ptr %91, align 4
  %92 = getelementptr ptr, ptr %48, i32 21
  store ptr %91, ptr %92, align 8
  %93 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 22
  store i64 3, ptr %93, align 4
  %94 = getelementptr ptr, ptr %48, i32 22
  store ptr %93, ptr %94, align 8
  %95 = getelementptr inbounds nuw %0, ptr %47, i32 0, i32 23
  store i64 1, ptr %95, align 4
  %96 = getelementptr ptr, ptr %48, i32 23
  store ptr %95, ptr %96, align 8
  %97 = load ptr, ptr @main1_kernel_module, align 8
  %98 = call ptr @mgpuModuleGetFunction(ptr %97, ptr @main1_kernel_main1_kernel_name)
  %99 = call ptr @mgpuStreamCreate()
  call void @mgpuLaunchKernel(ptr %98, i64 4, i64 3, i64 1, i64 1, i64 1, i64 1, i32 0, ptr %99, ptr %48, ptr null, i64 24)
  call void @mgpuStreamSynchronize(ptr %99)
  call void @mgpuStreamDestroy(ptr %99)
  %100 = call i32 @cudaMemcpy(ptr %6, ptr %34, i64 48, i32 4)
  %101 = call i32 @cudaFree(ptr %16)
  %102 = call i32 @cudaFree(ptr %25)
  %103 = call i32 @cudaFree(ptr %34)
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %13
}

define internal void @main1_kernel_load() section ".text.startup" {
entry:
  %0 = call ptr @mgpuModuleLoad(ptr @main1_kernel_binary, i64 5144)
  store ptr %0, ptr @main1_kernel_module, align 8
  ret void
}

declare ptr @mgpuModuleLoad(ptr, i64)

define internal void @main1_kernel_unload() section ".text.startup" {
entry:
  %0 = load ptr, ptr @main1_kernel_module, align 8
  call void @mgpuModuleUnload(ptr %0)
  ret void
}

declare void @mgpuModuleUnload(ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

declare ptr @mgpuModuleGetFunction(ptr, ptr)

declare ptr @mgpuStreamCreate()

declare void @mgpuLaunchKernel(ptr, i64, i64, i64, i64, i64, i64, i32, ptr, ptr, ptr, i64)

declare void @mgpuStreamSynchronize(ptr)

declare void @mgpuStreamDestroy(ptr)

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
