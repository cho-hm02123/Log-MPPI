classdef LogMPPIController < handle

    properties

        lambda                      % Temperature
        horizon                     % Prediction time
        n_samples                   % Number ouf rollouts
        cov                         % Covariance
        R                           % Control weight matix
        nu                          % Exploration variance

        model                       % Dynamics
        dt                          % Period

        obstacles

        control_sequence
        optimized_control_sequence

        goal
        state
        n_states = 5;               % Dimension of system's state [x, y, phi, vel, steer]

        rollouts_states             % Saving trajectory
        rollouts_costs              % Saving cost
        rollouts_plot_handle = [];   % Saving handle for visualization

        % constraints
        max_vel = 5;
        max_steer = 0.5;
        
        % Robot physical parameters for collision detection
        robot_radius = 0.1;         % Robot radius for collision detection

        % Log-MPPI specific parameters
        sigma_n = [0.002, 0.0022];  % Normal distribution variance for NLN mixture
        mu_ln = [0, 0];             % Log-normal mean (computed automatically)
        sigma_ln = [0, 0];          % Log-normal variance (computed automatically)
    end

    methods
        % Class Constructor
        function self = LogMPPIController(lambda, cov, nu, R, horizon, n_samples, model, dt, goal, obstacles)
            self.lambda = lambda;
            self.cov = cov;
            self.nu = nu;
            self.R = R;
            self.horizon = horizon;
            self.n_samples = n_samples;
            self.model = model;
            self.dt = dt;
            self.goal= goal;
            self.obstacles = obstacles;

            self.control_sequence = zeros(2,self.horizon);      % velocity, steer
            self.init_log_mppi_params();
        end


        function init_log_mppi_params(self)
            % Initialize parameters for Normal Log-Normal mixture
            % distribution
            % Based on equations (6) and (8) from the paper

            self.mu_ln = zeros(1, 2);
            self.sigma_ln = zeros(1, 2);

            % For each control dimension
            for i=1:2
                % Calculate mu_ln and sigma_ln for log-normal distribution
                % E(Y) such that mu_n * E(Y) gives desired behavior
                % Setting mu_ln such that E(Y) = exp(mu_ln + 0.5 * sigma_ln^2)
                self.mu_ln(i) = log(self.cov(i) / self.sigma_n(i)) - 0.5* (self.sigma_n(i))^2;
                self.sigma_ln(i) = sqrt(self.sigma_n(i));

                if self.mu_ln(i) < 0
                    self.mu_ln(i) = 0.1;
                end
                if self.sigma_ln(i) < 0.01
                    self.sigma_ln(i) = 0.1;
                end
            end
                fprintf('Log-MPPI initialized with mu_ln=[%.3f, %.3f], sigma_ln=[%.3f, %.3f]\n', ...
                    self.mu_ln(1), self.mu_ln(2), self.sigma_ln(1), self.sigma_ln(2));
        end


        function samples = sample_nln_mixture(self, n_samples, horizon)
            % Sample from Normal Log-Normal mixture MPPI distribution
            % Z = X + Y where X ~ N(0, sigma_n^2), Y ~ LN(mu_ln, sigma_ln^2)

            samples = zeros(2, n_samples, horizon);

            for dim =1:2
                % Sample from normal distribution X ~ N(0, sigma_n^2)
                X = normrnd(0, self.sigma_n(dim), [n_samples, horizon]);

                % Sample from log-normal distribution Y ~ LN(mu_ln, sigma_ln^2)
                Y = lognrnd(self.mu_ln(dim), self.sigma_ln(dim), [n_samples, horizon]);

                % Create NLN mixture Z = X * Y
                samples(dim, :,:) = X .* Y;
            end
        end


        function action = get_action(self, state)
            % Iniit variables
            self.state = state;
            init_state = state;
            states = zeros(self.n_states, self.horizon+1);
            S = zeros(self.n_samples, 1);

            self.rollouts_states = zeros(self.n_samples, self.horizon+1, self.n_states);
            self.rollouts_costs = zeros(self.n_samples, 1);

            % Generate control input disturbance
            delta_u_samples = self.sample_nln_mixture(self.n_samples, self.horizon);
            delta_u = delta_u_samples;

            % Apply constraints to delta_u for steering
            delta_u(2, delta_u(2,:,:) > 0.5) = 0.5;
            delta_u(2, delta_u(2,:,:) < -0.5) = -0.5;

            for k = 1:self.n_samples
                states(:,1) = init_state;

                for i = 1:self.horizon
                    % Single trajectory step
                    states(:,i+1) = self.model.step(self.control_sequence(:,i) + delta_u(:, k, i), self.dt, states(:,i));

                    % Compute cost of the state
                    S(k) = S(k) + self.cost_function(states(:,i+1), self.control_sequence(:,i),delta_u(:,k,i));

                end

                self.rollouts_states(k,:,:) = states';
                self.rollouts_costs(k) = S(k);
            end

            % Update the control according to the expectation over K sample trajectories
            S_normalized = S - min(S);
            for i=1:self.horizon
                self.control_sequence(:,i) = self.control_sequence(:,i) + self.total_entropy(delta_u(:,:,i)', S_normalized(:))';
            end

            % Output saturation
            self.control_sequence(1, self.control_sequence(1,:) > self.max_vel) = self.max_vel;
            self.control_sequence(1, self.control_sequence(1,:) < -self.max_vel) = -self.max_vel;

            self.control_sequence(2, self.control_sequence(2,:) > self.max_steer) = self.max_steer;
            self.control_sequence(2, self.control_sequence(2,:) < -self.max_steer) = -self.max_steer;

            % Select control action
            self.optimized_control_sequence = self.control_sequence;

            action = self.control_sequence(:,1);
            self.control_sequence = [self.control_sequence(:,2:end), [0;0]];
        end


        function cost = cost_function(self, state, u, du)
            state_cost = self.state_cost_function(state);
            control_cost = self.control_cost_function(u, du);

            cost = state_cost + control_cost;
        end


        function cost = state_cost_function(self,state)
            obstacle_cost = self.obstacle_cost_function(state);
            heading_cost = self.heading_cost_function(state);
            distance_cost = self.distance_cost_function(state);

            cost = obstacle_cost + heading_cost + distance_cost;
        end


        function cost = distance_cost_function(self, state)
            weight = 100;
            cost = weight*(self.goal(1:2) - state(1:2)')*(self.goal(1:2) - state(1:2)')';
        end


        function cost = heading_cost_function(self, state)
            weight = 100;
            pow = 2;
            cost = weight*abs(self.get_angle_diff(self.goal(3), state(3)))^pow;
        end


        function cost = control_cost_function(self, u, du)
            cost = (1-1/self.nu)/2 * du'*self.R*du + u'*self.R*du + 1/2*u'*self.R*u;
        end


        function [obstacle_cost] = obstacle_cost_function(self, state)
            if isempty(self.obstacles)
                obstacle_cost = 0;
                return
            end

            distance_to_obstacle = sqrt(sum((state(1:2)' - self.obstacles(:,[1:2])).^2,2));
            [min_dist, min_dist_idx] = min(distance_to_obstacle);

            % Consider robot radius for collision detection
            collision_threshold = self.obstacles(min_dist_idx,3) + self.robot_radius;
            
            if(min_dist <= collision_threshold)
                hit = 1;
            else
                hit = 0;
            end

            obstacle_cost = 750*exp(-min_dist/5) + 1e6*hit;
            % obstacle_cost = 550*exp(-min_dist/10) + 1e5*hit;
        end
        
        
        function collision = check_collision_with_obstacles(self, state)
            collision = false;
            if isempty(self.obstacles)
                return;
            end
            
            robot_position = state(1:2)';
            distances = sqrt(sum((robot_position - self.obstacles(:,1:2)).^2, 2));
            
            % Check if robot is within any obstacle radius (considering robot size)
            collision_thresholds = self.obstacles(:,3) + self.robot_radius;
            collision = any(distances <= collision_thresholds);
        end


        function value = total_entropy(self, du, trajectory_cost)
            exponents = exp(-1/self.lambda * trajectory_cost);

            value = sum(exponents.*du ./ sum(exponents), 1);
        end


         function plot_rollouts(self, fig)
            if ~isempty(self.rollouts_plot_handle)
                for i = 1:length(self.rollouts_plot_handle)
                    delete(self.rollouts_plot_handle(i));
                end
                self.rollouts_plot_handle = [];
            end
            figure(fig)
            costs = (self.rollouts_costs - min(self.rollouts_costs))/(max(self.rollouts_costs) - min(self.rollouts_costs));
            [~, min_idx] = min(costs);
            for i = 1:self.n_samples
                if i == min_idx
                    color = [0, 1, 1];
                else
                    color = [1-costs(i), 0, 0.2];
                end
                self.rollouts_plot_handle(end+1) = plot(self.rollouts_states(i,:,1), self.rollouts_states(i,:,2),'-', 'Color', color);
            end

            % Rollout of selected trajectory
            states = zeros(self.n_states, self.horizon+1);
            states(:,1) = self.state;

            for i = 1:self.horizon                    
                % Single trajectory step
                states(:,i+1) = self.model.step(self.optimized_control_sequence(:,i), self.dt, states(:,i));
            end
            self.rollouts_plot_handle(end+1) = plot(states(1,:), states(2,:), '--', 'Color', [0,1,0]);

        end     
    end

    methods(Static)
        function angle = get_angle_diff(angle1, angle2)
            angle_diff = angle1-angle2;
            angle = mod(angle_diff+pi, 2*pi) - pi;              
        end
    end

end