function f=makeTSUGFfilters
  deltaTheta = 45;
  orientation = 0:deltaTheta:(180-deltaTheta);
  
  f=zeros(47, 47, length(orientation));
  for o=1:length(orientation),
    f(:,:,o) = gabor(2*sqrt(2),orientation(o));
    f(:,:,o) = real(f(:,:,o))
  end
return

function h=gabor(wavelength, orientation)
  phi = 0;
  
  SpatialFrequencyBandwidth = 1.0;
  SpatialAspectRatio = 0.5
  sigmax = wavelength/pi*sqrt(log(2)/2)*(2^SpatialFrequencyBandwidth+1)/(2^SpatialFrequencyBandwidth-1);
  sigmay = sigmax./SpatialAspectRatio
  
  rx = ceil(7*sigmax)
  ry = ceil(7*sigmay)
  r = max(rx, ry)
  
  [X,Y] = meshgrid(-r:r,-r:r)
  
  Xprime = X .*cosd(orientation) - Y .*sind(orientation);
  Yprime = X .*sind(orientation) + Y .*cosd(orientation);
  
  hGaussian = exp( -1/2*( Xprime.^2 ./ sigmax^2 + Yprime.^2 ./ sigmay^2));
  hGaborEven = hGaussian.*cos(2*pi.*Xprime ./ wavelength+phi);
  hGaborOdd  = hGaussian.*sin(2*pi.*Xprime ./ wavelength+phi);
            
   h = complex(hGaborEven,hGaborOdd);
return
