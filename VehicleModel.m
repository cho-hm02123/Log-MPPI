classdef VehicleModel < handle

    properties
        m = 2000.0;               % mass [kg]
        tau_steering = 2;       % steering tme constant
        tau_velocity = 3;       % velocity time constant
        max_vel = 5;            % Max velocity [m/s]
    end

    methods
        function self = VehicleModel()
        end

        function state = step(self, action, dt, state)
            x = state(1);
            y = state(2);
            phi = state(3);
            prev_vel = state(4);
            prev_steer = state(5);

            vel = action(1);
            steer = action(2);

            vel = prev_vel + dt*(vel - prev_vel)/self.tau_velocity;
            steer = prev_steer + dt*(steer - prev_steer)/self.tau_steering;

            if vel > self.max_vel
                vel = self.max_vel;
            end

            phi = phi + steer;
            % phi = phi + vel*dt*tan(steer)/2.5;

            % - Kinematics Model of a Differential drive robot - %
            %        _      _     _            _   _   _         %
            %       |   x'   |   |  cos(phi) 0  | |  v  |        %
            %       |   y'   | = |  sin(phi) 0  | |_ w _|        %
            %       |_ phi' _|   |_   0      1 _|                %
            %                                                    %
            % -------------------------------------------------- %

            x = x + vel*dt*cos(phi);
            y = y + vel*dt*sin(phi);

            state = [x, y, phi, vel, steer];
        end
    end
end