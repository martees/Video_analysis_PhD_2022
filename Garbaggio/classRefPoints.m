% APE, Oct 14, 2022

classdef classRefPoints
    properties        
        xy_holes % x,y coordinates of the 4 reference holes 
        side_square_mm % Length of the side of the square of reference points, in mm
        quality_square % Quality of the square formed by the 4 reference points        
        pars % Parameters about the reference holes
        transf % Information needed to transform between the image (in pixels) and the aligned reference frame (in mm, rotated and with the origin at the center of the plate)        
        error % Information about errors        
        aux % For auxiliary data I may need
    end % properties    
    methods
        %% Constructor
        function refPoints = classRefPoints(xy_holes, varargin)
            th_quality_square = .05; % If reference square has quality above this threshold, the plate will be rejected
            side_square_mm = 1; % If we have no information about the side, we take 1 mm (which is ridiculuous; just to put something)
            parseinputs
            
            refPoints.xy_holes = xy_holes;
            refPoints.side_square_mm = side_square_mm;
            refPoints.pars.th_quality_square = th_quality_square;  
            refPoints.error.errorList = [];
            refPoints.error.errorMeaningList = {};
            if size(refPoints.xy_holes, 1) == 4
                refPoints.quality_square = posRef2qualitySquare(refPoints.xy_holes);
                [x0,v1,v2] = posRef2base(refPoints.xy_holes); % x0 is at the lower-left point, and v1, v2 have length equal to the square side
                % Translate base to center of plate, and renormalize so that
                % vectors have length 1 mm
                x0 = x0 + v1/2 + v2/2;
                mmPerPixel = refPoints.side_square_mm/norm(v1);
                v1 = v1/norm(v1);
                v2 = v2/norm(v2);
                refPoints.transf.x0 = x0; %lower left corner
                refPoints.transf.v1 = v1; % unit vectors
                refPoints.transf.v2 = v2;                
                refPoints.transf.mmPerPixel = mmPerPixel; % scaling factor                                   
                if refPoints.quality_square > refPoints.pars.th_quality_square
                    refPoints = refPoints.addError(1.2, 'Reference points not usable: Quality not good enough', 'show', true);
                end
            else
                refPoints = refPoints.addError(1, 'Reference points not usable: They are not 4.', 'show', true);                
            end         
        end % constructor
        
        %% pixel2mm: Transforms a set of points, from the image reference system (in pixels) into the aligned ref. system (in mm)
        function xy = pixel2mm(refPoints, xy)
            xy = xy - refPoints.transf.x0;
             xy = xy*refPoints.transf.mmPerPixel;
        end % function pixel2mm
        
        %% mm2pixel: Transforms a set of points, from the aligned ref. system (in mm) into the image reference system (in pixels)
        function xy = mm2pixel(refPoints, xy)
            xy = xy/refPoints.transf.mmPerPixel;
            xy = xy(:, 1).*refPoints.transf.v1 + xy(:, 2).*refPoints.transf.v2;
            xy = xy + refPoints.transf.x0;
        end % function mm2pixel
        
        %% addError: Adds an error
        function plate = addError(plate,codeError,strError,varargin)
            show = false;
            parseinputs
            plate.error.errorList(end+1) = codeError;
            plate.error.errorMeaningList{end+1} = strError;
            if show
                disp(strError)
            end
        end % function addError
    end % methods
end % class

%% AUXILIARY FUNCTIONS

%% posRef2base
function [x0,v1,v2] = posRef2base(posRef,varargin)
% I think this function gives you the two perpendicular vectors (v1 and v2) 
% that best approximate the 4 reference points.
show = false;
parseinputs

% Reorder points
[~,orderY] = sort(posRef(:,2));
orderFinal = NaN(1,4);
for iRow = 1:2
    indPointRow = orderY((iRow-1)*2 + (1:2));
    [~,orderX] = sort(posRef(indPointRow,1));
    orderFinal((iRow-1)*2 + (1:2)) = indPointRow(orderX);
end % iRow
posRef = posRef(orderFinal,:);

% Initial estimate for base, using only the two bottom points
x0 = posRef(1,:);
v1 = diff(posRef(1:2,:),1,1);


% Minimization of error
x0v1 = [x0 v1];
x0v1 = fminsearch(@(x) funError(x(1:2),x(3:4),posRef),x0v1);
x0 = x0v1(1:2);
v1 = x0v1(3:4);
v2 = [-v1(2) v1(1)];

if show
    figure
    plot(posRef(:,1),posRef(:,2),'.')
    hold on
    posTheor = [x0 ; x0+v1 ; x0+v2 ; x0+v1+v2];
    plot(posTheor(:,1),posTheor(:,2),'o')
    axis image
end % if show

end

%% Error function for posRef2base
function err = funError(x0,v1,posRef)
    v2 = [-v1(2) v1(1)];
    err = sum((posRef(1,:)-x0).^2 + (posRef(2,:) - (x0+v1)).^2 + (posRef(3,:) - (x0+v2)).^2 + (posRef(4,:) - (x0+v1+v2)).^2);
end

function qualitySquare = posRef2qualitySquare(posRef)
% I think this function takes the coordinates of the 4 reference points (in
% posRef) and gives you a number that is 1 if they form a perfect square,
% and lower if its a shitty square.
qualitySquare = 0;
distMat = sum((permute(posRef,[1 3 2]) - permute(posRef,[3 1 2])).^2,3);
for iPoint = 1:4
    % Vectors with the two nearest points
    [~,order] = sort(distMat(iPoint,:));
    vec1 = posRef(iPoint,:) - posRef(order(2),:);
    vec1 = vec1/norm(vec1);
    vec2 = posRef(iPoint,:) - posRef(order(3),:);
    vec2 = vec2/norm(vec2);
    qualitySquare = qualitySquare + abs(vec1*vec2');
end
end % function qualitySquare