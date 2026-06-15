#/cnl/xaos/julia-1.12.6/bin/julia -t auto

MacSalk="SALK";
pfad_home="/home/claudia";

include("DDAsetup.jl");
include("DDAfunctions.jl");

#import Pkg; Pkg.add("DelayDifferentialAnalysis")
#=
import Pkg; Pkg.update("DelayDifferentialAnalysis")
=#
using DelayDifferentialAnalysis

N_MOD=3; nr_delays=2; DDAorder=4;
(MOD,P_DDA,SSYM) = make_MOD_new_new(N_MOD,nr_delays,DDAorder);

f=findall(sum(MOD[:,1:2],dims=2) .== 2); f = map(x -> x[1],f);
MOD = MOD[f,:]; SSYM = SSYM[f,:];

TM=50; dm=4; 
make_TAU_ALL(SSYM,dm+1:TM);

WL = 1024; #WS = 256; #SR = [500; 1024];

#NrCH= 2;WS = 60*WL;
NrCH=10;WS = 20*WL;


DDA_DIR = @sprintf("DDA_EEG_%02d",NrCH);; dir_exist(DDA_DIR);


FILES = filter(endswith(".edf"),readdir("EEG_Yujie"));

for n_file=scramble_list(1:length(FILES))
  FN_data = @sprintf("EEG_Yujie%s%s",SL,FILES[n_file]);
  for mm = scramble_list(1:size(MOD,1))
    (MODEL,SYM,model,L_AF) = make_MODEL_new(MOD,SSYM,mm);

    FN_DDA = @sprintf("%s%s%s__%s",DDA_DIR,SL,model,replace(FILES[n_file], ".edf" => ".DDA"));
    TAU_name = @sprintf("TAU_ALL__%d_%d",SYM[1],SYM[2]);
    TAU = readdlm(TAU_name); N_TAU=size(TAU,1);

    R = scramble_list(1:21); CH=sort(R[1:10]);
    #CH = 1:21;

    if !isfile(join([FN_DDA,"_ST"]))
       touch(join([FN_DDA,"_ST"]));

       run_DDA(
          file_path=FN_data,
          channels=CH,          
          flavors=["ST"],
          out_fn = FN_DDA,  
          binary_path="/home/claudia/TOOLS/run_DDA_AsciiEdf",
          model=MODEL, 
          WL=WL,
          WS=WS,
          derivative_points=dm,                
          order=DDAorder,  
          tau_file=TAU_name,
          #load_results=false
         );
    end;
  end
end



#=
/cnl/xaos/julia-1.12.6/bin/julia -t auto BestModel_EEG.jl > & /dev/null &
/cnl/xaos/julia-1.12.6/bin/julia -t auto BestModel_EEG.jl > & /dev/null &
/cnl/xaos/julia-1.12.6/bin/julia -t auto BestModel_EEG.jl > & /dev/null &
/cnl/xaos/julia-1.12.6/bin/julia -t auto BestModel_EEG.jl > & /dev/null &
/cnl/xaos/julia-1.12.6/bin/julia -t auto BestModel_EEG.jl > & /dev/null &
/cnl/xaos/julia-1.12.6/bin/julia -t auto BestModel_EEG.jl > & /dev/null &
/cnl/xaos/julia-1.12.6/bin/julia -t auto BestModel_EEG.jl > & /dev/null &
/cnl/xaos/julia-1.12.6/bin/julia -t auto BestModel_EEG.jl > & /dev/null &
/cnl/xaos/julia-1.12.6/bin/julia -t auto BestModel_EEG.jl > & /dev/null &
/cnl/xaos/julia-1.12.6/bin/julia -t auto BestModel_EEG.jl > & /dev/null &
=#


