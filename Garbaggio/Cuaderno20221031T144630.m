%% Add patch conditions to mini-patch videos
% Name of this file: Cuaderno20221031T144630.m
%
%

pwd

%% Define folders
clear all
general_folder = 'H:\Results_minipatches_20221108';
folders_dir = dir(general_folder);
folders_dir = folders_dir(3:end);
for i_folder = length(folders_dir):-1:1
    folders{i_folder} = fullfile(general_folder, folders_dir(i_folder).name);
end

%% Define conditions (from file:///K:/RLlC20221012_small_drops.html)
distantSpaceList = [
    [-9.0, -15.59], 
    [9.0, -15.59], 
    [-18.0, 0.0], 
    [0.0, 0.0], 
    [18.0, 0.0], 
    [-9.0, 15.59], 
    [9.0, 15.59]];

mediumSpaceList = [
    [-13.5, -15.59], 
    [-4.5, -15.59], 
    [4.5, -15.59], 
    [13.5, -15.59], 
    [-18.0, -7.79], 
    [-9.0, -7.79], 
    [0.0, -7.79], 
    [9.0, -7.79], 
    [18.0, -7.79],
    [-22.5, 0.0], 
    [-13.5, 0.0], 
    [-4.5, 0.0], 
    [4.5, 0.0], 
    [13.5, 0.0], 
    [22.5, 0.0],
    [-18.0, 7.79], 
    [-9.0, 7.79], 
    [0.0, 7.79], 
    [9.0, 7.79], 
    [18.0, 7.79], 
    [-13.5, 15.59], 
    [-4.5, 15.59], 
    [4.5, 15.59], 
    [13.5, 15.59]];

alpha = -15/180*pi;
mediumSpaceListOrig = mediumSpaceList;
for iPatch = 1:length(mediumSpaceList)
    xy = mediumSpaceList(iPatch, :);
    mediumSpaceList(iPatch, :) = [xy(1)*cos(alpha) - xy(2)*sin(alpha), xy(1)*sin(alpha) + xy(2)*cos(alpha)];
end

closeSpaceList = [
    [-15.75, -11.69], 
    [-11.25, -11.69], 
    [-6.75, -11.69], 
    [-2.25, -11.69], 
    [2.25, -11.69], 
    [6.75, -11.69], 
    [11.25, -11.69], 
    [15.75, -11.69], 
    [-13.5, -7.79], 
    [-9.0, -7.79], 
    [-4.5, -7.79], 
    [0.0, -7.79], 
    [4.5, -7.79], 
    [9.0, -7.79], 
    [13.5, -7.79], 
    [-15.75, -3.90], 
    [-11.25, -3.90], 
    [-6.75, -3.90], 
    [-2.25, -3.90], 
    [2.25, -3.90], 
    [6.75, -3.90], 
    [11.25, -3.90], 
    [15.75, -3.90], 
    [-13.5, 0.0], 
    [-9.0, 0.0], 
    [-4.5, 0.0], 
    [4.5, 0.0], 
    [9.0, 0.0], 
    [13.5, 0.0], 
    [-15.75, 3.90], 
    [-11.25, 3.90], 
    [-6.75, 3.90], 
    [-2.25, 3.90], 
    [2.25, 3.90], 
    [6.75, 3.90], 
    [11.25, 3.90], 
    [15.75, 3.90], 
    [-13.5, 7.79], 
    [-9.0, 7.79], 
    [-4.5, 7.79], 
    [0.0, 7.79], 
    [4.5, 7.79], 
    [9.0, 7.79], 
    [13.5, 7.79], 
    [-15.75, 11.69], 
    [-11.25, 11.69], 
    [-6.75, 11.69], 
    [-2.25, 11.69], 
    [2.25, 11.69], 
    [6.75, 11.69], 
    [11.25, 11.69], 
    [15.75, 11.69], 
    ];
clusterList = [
    [-16.14, -9.6], 
    [-12.1, -9.96], 
    [-18.94, -6.54], 
    [-14.48, -5.68], 
    [-20.09, -11.42], 
    [-2.71, 12.51], 
    [-4.6, 7.81], 
    [0.47, 15.37], 
    [0.65, 10.3], 
    [-6.82, 11.76], 
    [5.97, -16.54], 
    [7.33, -11.5], 
    [9.23, -18.94], 
    [2.66, -14.34], 
    [5.56, -20.51], 
    [17.69, 4.04], 
    [15.78, 8.05], 
    [20.36, 6.92], 
    [13.3, 3.47], 
    [16.1, 0.44], 
    [-19.21, 8.89], 
    [-22.63, 6.53], 
    [-15.15, 8.95], 
    [-18.35, 5.02]];

% clusterList = [[-12.42, -7.39], % OLD positions used on the 10th
%     [-9.31, -7.04], 
%     [-14.99, -5.5], 
%     [-11.77, -4.18], 
%     [-15.11, -9.36], 
%     [-2.08, 9.63], 
%     [0.18, 6.44], 
%     [-2.61, 12.87], 
%     [0.75, 10.87], 
%     [-3.31, 6.65], 
%     [5.36, -14.26], 
%     [2.2, -16.74], 
%     [4.45, -11.28], 
%     [6.38, -17.14], 
%     [7.44, -12], 
%     [13.61, 3.11], 
%     [12.14, 6.19], 
%     [15.67, 5.32], 
%     [10.23, 2.67], 
%     [12.39, 0.34], 
%     [-16.32, 7.61], 
%     [-16.9, 4.47], 
%     [-14.13, 9.85], 
%     [-13.75, 5.96]];

mediumSpaceHighDensityMask = [0, 1, 0, 1, 0, 1, 0 , 1, 0, 0, 1, 0 , 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1];

% Centers of patches for each condition
cond2centers = {
    closeSpaceList 
    mediumSpaceList
    distantSpaceList
    clusterList
    closeSpaceList 
    mediumSpaceList
    distantSpaceList
    clusterList
    mediumSpaceList
    mediumSpaceList
    mediumSpaceList
    mediumSpaceList};

for i = 1:length(cond2centers)
    cond2centers{i} = -cond2centers{i}; % Invert the y coordinate because images have y axis reversed in matlab. Invert x coordinate because were looking from below.
end

% Densities for each condition
cond2densities = {};
i_cond = 0;
for density = [.2 .5]
    for i_grid = 1:4
        i_cond = i_cond + 1;
        cond2densities{i_cond} = density*ones(size(cond2centers{i_cond}, 1), 1);
    end % i_grid
end % density
i_cond = i_cond + 1;
cond2densities{i_cond} = 1.25*ones(size(cond2centers{i_cond}, 1), 1);
i_cond = i_cond + 1;
density0 = .2;
density1 = .5;
cond2densities{i_cond} = density0*ones(size(cond2centers{i_cond}, 1), 1);
cond2densities{i_cond}(mediumSpaceHighDensityMask == 1) = density1;
i_cond = i_cond + 1;
density0 = 1.25;
density1 = .5;
cond2densities{i_cond} = density0*ones(size(cond2centers{i_cond}, 1), 1);
cond2densities{i_cond}(mediumSpaceHighDensityMask == 1) = density1;
i_cond = i_cond + 1;
cond2densities{i_cond} = 0*ones(size(cond2centers{i_cond}, 1), 1);

%% Add the info to each video
for i_folder = 1:length(folders)
    last_sep = find(folders{i_folder} == '\'', 1, 'last');
    date_exp = folders{i_folder}(last_sep + 1:last_sep + 8);
    if ~strcmpi(date_exp, '20221007') % Exclude the pilot study
        if strcmpi(date_exp, '20221011') % Day when we were still using the 3x3 code (it corresponds to the robot script of 20221010)            
            rowcol_code = [3 3]; % Number of rows and columns of the code
        else
            rowcol_code = [2 4]; % Number of rows and columns of the code
        end
        if (exist(char(fullfile(folders{i_folder}, 'holes.mat')), 'file') == 0)
            disp([char(fullfile(folders{i_folder})) ' is missing the holes'])
        else
            load(char(fullfile(folders{i_folder}, 'holes.mat')))
            if strcmp(folders{i_folder}(end-35:end), '20221012T200743_SmallPatches_C2-CAM6')
                pointList(end, 3) = 2; % Correct a mistake manually
            end
            if ~any(pointList(:, 3) == 4)
                refPoints = classRefPoints(pointList(pointList(:, 3) == 2, 1:2), 'side_square_mm', 32);
                if isempty(refPoints.error.errorList)
                    xy_code = refPoints.pixel2mm(pointList(pointList(:, 3) == 1, 1:2));
                    %                 code = classCode(xy_code, 'n_row_col', [3 3]);
                    code = classCode(xy_code, 'n_row_col', rowcol_code);
                    if isempty(code.error.errorList) && (~strcmpi(date_exp, '20221011') || (code.num ~= 3 && code.num ~= 7)) % Exclude the clusters from the 11th (they had a different arrangement) NOTE: That day we played with two different volumes, marked with x on the plates. We ll need to exclude manually part of the plates (or remove the whole day)
                        centers_patches = cond2centers{code.num + 1};
                        centers_patches = refPoints.mm2pixel(centers_patches); % Transform to the image reference frame
                        densities_patches = cond2densities{code.num + 1};
                        num_condition = code.num;
                        save(char(fullfile(folders{i_folder}, 'foodpatches.mat')), 'centers_patches', 'densities_patches', 'num_condition', 'rowcol_code')
                        disp([char(fullfile(folders(i_folder))) ' successful'])
                        if (code.num == 3 || code.num == 7) && strcmpi(date_exp, '20221011')
                            figure
                            load(char(fullfile(folders{i_folder}, 'composite_patches.mat')))
                            imagesc(composite_patches)
                            colormap gray
                            axis equal
                            hold on
                            plot(pointList(:, 1), pointList(:, 2), 'o')
                            plot(centers_patches(:, 1), centers_patches(:, 2), 'r.')
                            title(folders{i_folder})
                        end
                    end % if no errors in the code
                else
                    fprintf('FAILED: %s.\n', refPoints.error.errorMeaningList{1})
                end % if no errors in the reference holes
            else
                disp([char(fullfile(folders(i_folder))) ' was marked as not valid'])
            end
        end
    end % if not pilot
end % i_folder

%%
% load('K:\TestDrops\CAM2\holes.mat')
% refPoints = classRefPoints(pointList(pointList(:, 3) == 2, 1:2), 'side_square_mm', 32);
% xy_code = refPoints.pixel2mm(pointList(pointList(:, 3) == 1, 1:2));
% code = classCode(xy_code, 'n_row_col', [2 4]);
% 
% centers_patches = cond2centers{code.num};
% n_patch = size(centers_patches, 1);
% clear fp
% r_patch = 1.5; % Approximate radius of the food patches, in mm
% manualpoints = [];
% for i_patch = n_patch:-1:1
%     % Four points along the edge of the food patch
%     xy_edge = centers_patches(i_patch, :) + r_patch*[1 0 ; 0 1 ; -1 0 ; 0 -1];
%     xy_edge = refPoints.mm2pixel(xy_edge); % Transform to the image reference frame
%     manualpoints = [manualpoints ; zeros([4 1]) xy_edge repmat(i_patch, [4 1])];
%     fp(i_patch) = foodpatch(xy_edge);
% end

% %% Check alignment
% clear vt
% for i_video = 1:size(folders, 2)
%     folders_cam = dir(fullfile(folders{i_video, 1}, 'CAM*'));
%     for i_cam = 1:length(folders_cam)
%         for i_part = 1:2
%             vt(i_video, i_cam, i_part) = videotracker(fullfile(folders{i_video, i_part}, folders_cam(i_cam).name, 'Video1.tif'));
%             figure
%             imagesc(vt(i_video, i_cam, i_part).frame(round(vt(i_video, i_cam, i_part).n_frames/2)))
%             colormap gray
%             axis image
%         end % i_part
%         caca
%     end % i_cam
%     caca
% end % i_video

%% Check alignment of the food patches

% for i_video = 1:size(folders, 2)
%     folders_cam = dir(fullfile(folders{i_video, 1}, 'CAM*'));
%     for i_cam = 1:length(folders_cam)
%         %         if exist(fullfile(folders{i_video, 1}, folders_cam(i_cam).name, 'Tracking_Video', 'fp.mat'), 'file')
%         %             disp(['Food patches already traced for ' fullfile(folders{i_video, 1}, folders_cam(i_cam).name)])
%         %         else
%         if exist(fullfile(folders{i_video, 1}, folders_cam(i_cam).name, 'foodpatches.mat'), 'file') == 2
%             load(fullfile(folders{i_video, 1}, folders_cam(i_cam).name, 'foodpatches.mat'))
% %             load(fullfile(folders{i_video, 1}, folders_cam(i_cam).name, 'holes.mat'))
% %             refPoints = classRefPoints(pointList(pointList(:, 3) == 2, 1:2), 'side_square_mm', 32);
% %             xy_code = refPoints.pixel2mm(pointList(pointList(:, 3) == 1, 1:2));
% %             code = classCode(xy_code, 'n_row_col', [2 4]);
% %             
% %             centers_patches = cond2centers{code.num + 1}; % +1 because Python starts at 0
% %             n_patch = size(centers_patches, 1);
% %             clear fp
% %             r_patch = 1.5; % Approximate radius of the food patches, in mm
% %             manualpoints = [];
% %             for i_patch = n_patch:-1:1
% %                 % Four points along the edge of the food patch
% %                 xy_edge = centers_patches(i_patch, :) + r_patch*[1 0 ; 0 1 ; -1 0 ; 0 -1];
% %                 xy_edge = refPoints.mm2pixel(xy_edge); % Transform to the image reference frame
% %                 manualpoints = [manualpoints ; zeros([4 1]) xy_edge repmat(i_patch, [4 1])];
% %                 fp(i_patch) = foodpatch(xy_edge);
% %             end
%             load(fullfile(folders{i_video, 1}, folders_cam(i_cam).name, 'Tracking_Video', 'vt.mat'))
%             disp(['Checking food patches for ' fullfile(folders{i_video, 1}, folders_cam(i_cam).name) '...'])
% %             vt.aux.manualpoints = manualpoints;
%             h=showvideo(vt,'tracingpatch',true);
%             hold on
%             plot(centers_patches(:, 1), centers_patches(:, 2), 'rx')
%             drawnow
%             maximize
%             uiwait(h.fig)
% %             clear fp
% %             for c_patches=1:max(vt.aux.manualpoints(:,4))
% %                 fp(c_patches)=foodpatch(vt.aux.manualpoints(vt.aux.manualpoints(:,4)==c_patches,2:3));
% %             end
% %             vt.save
% %             save(fullfile(vt.folder,'fp.mat'),'fp')
%             fprintf('Done\n')
%             %         end % i_part
%         else
%             disp(['No patch information for ' fullfile(folders{i_video, 1}, folders_cam(i_cam).name)])
%         end
%     end % i_cam
% end % i_video