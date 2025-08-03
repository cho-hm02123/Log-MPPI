%% Display initialization
close all
clear
clc

%% Algorithm Flow
%                 Initial State
%                       |
%                       V
%         Sampled control input sequence 
%                       |
%                       V
%       State Estimation using System model
%                       |
%                       V
%                 Computing cost
%                       |
%                       V
% Updating optimal input sequence based on softmin weight
%                       |
%                       V
%         Load optimal control input u_0
%         

%% Definite Param
n_rollout = 357;            % Number of rollout trajectories
horizon = 25;               % Prediction horizon presented as number of steps (2.5s x 10Hz)
lambda = 10;                % Temperature (alpha in paper) - Selectiveness of trajectories by cost 
nu = 500;                   % Exploration variance
R = diag([1,5]);            % Control weight matrix
cov = [1, 0.4];             % Variance of control inputs disturbance
% lambda = 10/3;                % Temperature (alpha in paper) - Selectiveness of trajectories by cost 
% nu = 1200;                   % Exploration variance
% R = lambda*diag([0.5,0.5])^(-1/2)            % Control weight matrix
% cov = diag([0.5,0.5]);           % Variance of control inputs disturbance
dt = 0.1;                   % Time step controller and simulation (10Hz)

init_pose = zeros(1,5);     % Initial pose [x,y,phi,v,steer]
goal_pose = [6,6,0];
goal_tolerance = 0.3;       % Distance tolerance for goal reaching

%% Setup Enviroment - Obstacles
% obstacles=[];
% 
% n_obstacles = 30;
% obstacles = [rand(n_obstacles,2)*4+1, 0.2*ones(n_obstacles,1)];  %(x,y,radias)
% size(obstacles)

o = load("ob1.mat");
obstacles = o.obstacles;
n_obstacles = size(obstacles);

%% Init Dynamics & Controller
v_dynamics = VehicleModel();
car = VehicleModel();
controller = LogMPPIController(lambda, cov, nu, R, horizon, n_rollout, car, dt, goal_pose, obstacles);

%% Visualization
fig = figure;

hold on
axis equal
xlim([-0.5 + min(init_pose(1), goal_pose(1)), 0.5 + max(init_pose(1), goal_pose(1))]);
ylim([-0.5 + min(init_pose(2), goal_pose(2)), 0.5 + max(init_pose(2), goal_pose(2))]);
plot_pose(init_pose, 'bo');
plot_pose(goal_pose, 'ro');
plot_obstacles(obstacles);

%% Data recording initialization
max_steps = 1000;
time_data = [];
input_velocity = [];
input_steering = [];
input_heading = [];
actual_velocity = [];
actual_steering = [];
actual_heading = [];
collision_times = [];
collision_positions = [];

%% Control
car_pose = init_pose;
goal_reached = false;
collision_detected = false;
step_count = 0;

for i = 1:max_steps
    step_count = i;
    current_time = (i-1) * dt;
    
    % Get control action
    action = controller.get_action(car_pose);
    
    % Record input data
    time_data(end+1) = current_time;
    input_velocity(end+1) = action(1);
    input_steering(end+1) = action(2);
    input_heading(end+1) = car_pose(3); % Current heading before action
    
    % Update vehicle state
    car_pose = v_dynamics.step(action, dt, car_pose);
    
    % Record actual data
    actual_velocity(end+1) = car_pose(4);
    actual_steering(end+1) = car_pose(5);
    actual_heading(end+1) = car_pose(3);
    
    % Check for collision
    if check_collision(car_pose, obstacles)
        collision_detected = true;
        collision_times(end+1) = current_time;
        collision_positions(end+1,:) = car_pose(1:2);
        fprintf('Collision detected at time %.2f at position (%.2f, %.2f)\n', current_time, car_pose(1), car_pose(2));
    end
    
    % Check if goal is reached
    distance_to_goal = norm(car_pose(1:2) - goal_pose(1:2));
    if distance_to_goal <= goal_tolerance
        goal_reached = true;
        fprintf('Goal reached at time %.2f! Distance to goal: %.3f\n', current_time, distance_to_goal);
        break;
    end
    
    % Visualization
    if mod(i,2) == 0
        controller.plot_rollouts(fig);
        exportgraphics(gcf, 'animation2.gif', 'Append', true);
        drawnow;
    end
    
    plot_pose(car_pose, 'go');
    
    % Safety check - if too far from goal, stop
    if distance_to_goal > 20
        fprintf('Robot moved too far from goal. Stopping simulation.\n');
        break;
    end
end

%% Final status
if goal_reached
    fprintf('Simulation completed successfully! Goal reached in %d steps (%.2f seconds).\n', step_count, step_count*dt);
else
    fprintf('Simulation ended without reaching goal. Final distance: %.3f\n', norm(car_pose(1:2) - goal_pose(1:2)));
end

if collision_detected
    fprintf('Total collisions detected: %d\n', length(collision_times));
end

%% Plot simulation results
plot_simulation_results(time_data, input_velocity, input_steering, input_heading, ...
                       actual_velocity, actual_steering, actual_heading);

%% Plot collision data if any collisions occurred
if ~isempty(collision_times)
    plot_collision_data(collision_times, collision_positions);
end

%% Utility functions
function plot_obstacles(obstacles)
    for i = 1:size(obstacles,1)
        r = obstacles(i,3);
        pos = [obstacles(i,[1,2])-r 2*r 2*r];

        rectangle('Position',pos, 'Curvature',[1,1], 'FaceColor','k','EdgeColor','none');
    end
end

function plot_pose(pose, style)
    x = pose(1);
    y = pose(2);
    phi = pose(3);

    plot(x,y,style);
    [delta_x, delta_y] = pol2cart(phi,0.5);
    quiver(x,y,delta_x,delta_y);
end

function collision = check_collision(pose, obstacles)
    collision = false;
    if isempty(obstacles)
        return;
    end
    
    robot_position = pose(1:2);
    distances = sqrt(sum((robot_position - obstacles(:,1:2)).^2, 2));
    min_distance = min(distances);
    
    % Check if robot is within any obstacle radius
    obstacle_radii = obstacles(:,3);
    collision = any(distances <= obstacle_radii);
end

function plot_simulation_results(time_data, input_vel, input_steer, input_heading, ...
                                actual_vel, actual_steer, actual_heading)
    figure('Name', 'Simulation Results', 'Position', [100, 100, 1200, 800]);
    
    % Convert steering to angular velocity (approximate)
    input_angular_vel = input_steer;  % For simple model
    actual_angular_vel = actual_steer;
    
    % Plot 1: Velocity comparison
    subplot(3,1,1);
    plot(time_data, input_vel, 'r-', 'LineWidth', 2, 'DisplayName', 'Input Velocity');
    hold on;
    plot(time_data, actual_vel, 'b--', 'LineWidth', 2, 'DisplayName', 'Actual Velocity');
    xlabel('Time (s)');
    ylabel('Velocity (m/s)');
    title('Velocity Comparison');
    legend('Location', 'best');
    grid on;
    
    % Plot 2: Angular velocity comparison
    subplot(3,1,2);
    plot(time_data, input_angular_vel, 'r-', 'LineWidth', 2, 'DisplayName', 'Input Angular Velocity');
    hold on;
    plot(time_data, actual_angular_vel, 'b--', 'LineWidth', 2, 'DisplayName', 'Actual Angular Velocity');
    xlabel('Time (s)');
    ylabel('Angular Velocity (rad/s)');
    title('Angular Velocity Comparison');
    legend('Location', 'best');
    grid on;
    
    % Plot 3: Heading comparison
    subplot(3,1,3);
    plot(time_data, input_heading, 'r-', 'LineWidth', 2, 'DisplayName', 'Input Heading');
    hold on;
    plot(time_data, actual_heading, 'b--', 'LineWidth', 2, 'DisplayName', 'Actual Heading');
    xlabel('Time (s)');
    ylabel('Heading (rad)');
    title('Heading Comparison');
    legend('Location', 'best');
    grid on;
    
    sgtitle('Robot Control Input vs Actual State', 'FontSize', 16, 'FontWeight', 'bold');
end

function plot_collision_data(collision_times, collision_positions)
    figure('Name', 'Collision Analysis', 'Position', [200, 200, 1000, 600]);
    
    % Plot collision times
    subplot(2,1,1);
    stem(collision_times, ones(size(collision_times)), 'r', 'filled', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Time (s)');
    ylabel('Collision Event');
    title('Collision Events Over Time');
    grid on;
    ylim([0, 1.5]);
    
    if ~isempty(collision_times)
        % Add text annotations for collision count
        for i = 1:length(collision_times)
            text(collision_times(i), 1.1, sprintf('C%d', i), 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        end
    end
    
    % Plot collision positions
    subplot(2,1,2);
    if ~isempty(collision_positions)
        plot(collision_times, collision_positions(:,1), 'ro-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'X Position');
        hold on;
        plot(collision_times, collision_positions(:,2), 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Y Position');
        xlabel('Time (s)');
        ylabel('Position (m)');
        title('Collision Positions Over Time');
        legend('Location', 'best');
        grid on;
    else
        text(0.5, 0.5, 'No Collisions Detected', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
        xlim([0, 1]);
        ylim([0, 1]);
    end
    
    sgtitle(sprintf('Collision Analysis - Total Collisions: %d', length(collision_times)), 'FontSize', 16, 'FontWeight', 'bold');
end