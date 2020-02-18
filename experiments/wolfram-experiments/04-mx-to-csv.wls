#!/home/pmd/wolfram/bin/wolframscript

begin=Now;
Print["Load MX data"]
<<"all-features.mx"
Print["Loading complete"]

(*
featuresOfAllPhotos//Head
featuresOfAllPhotos//Length
*)

Print["Export as CSV"]
Export["all-features.csv",featuresOfAllPhotos];
Print["Export complete"]
end=Now

Print["Operation took "<>ToString[end-begin]]

