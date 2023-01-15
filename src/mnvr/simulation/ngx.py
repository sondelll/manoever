import tensorflow as tf
import beamngpy
from beamngpy.sensors import AdvancedIMU, Camera, PowertrainSensor

from .processing import NGSight, NGVehicleSense
from PIL import Image


class ExpertNGSimulation:
    """Runner for executing episodes in the BeamNG Simulator.
    
    Make sure your mnvr.yaml file reflects the setup you have on your system.
    """
    def __init__(
        self,
        scenario:beamngpy.Scenario = None,
        start_pos:tuple = (0,0,0),
        start_rot:tuple = (0,0,0)
    ) -> None:
        import yaml
        from beamngpy import BeamNGpy, Vehicle, Scenario, angle_to_quat
        from ..io.yaml_read import MnvrConfig
        
        self.cfg = MnvrConfig()
        
        self.bng = self.cfg.get_prebaked_beamng()

        try:
            self.bng.open(launch=False)
        except:
            self.bng.open(launch=True)
        
        self.bng.change_setting('defaultGearboxBehavior', 'realistic')
        
        self.start_pos, self.start_rot = start_pos, start_rot
        self.target_scenario = scenario
        
        self.car = Vehicle(
            'scenario_player0',
            model='bastion',
            part_config=f"{self.cfg.user_folder}/0.27/vehicles/bastion/manoever_test_car.pc"
        )
        
        
    def run(
        self,
        truncate_seconds:float = 10.0,
        warm_up_steps:int = 3,
        sample_step_size:int = 2,
        shape_debug:bool = False
    ) -> tuple[
            list[tuple[tf.Tensor, tf.Tensor]],
            tuple[dict, dict, float]
        ]:
        """Do a training run in BeamNG.tech :)

        Args:
            agent (tf.keras.Model): Model that will receive data and return commands.
            truncate_seconds (float, optional): Simulation limit. Defaults to 10.0.
            warm_up_steps (int, optional): Number of steps to advance before attempting to call the agent. Defaults to 3.
            sample_step_size (int, optional): Number of steps between sampling state+action. Defaults to 5.

        Raises:
            RuntimeError: When stuff goes sideways.

        Returns:
            tuple[  
                samples -> list[tuple[tf.Tensor, tf.Tensor]],  
                simulation data -> tuple[dict, dict, float]  
                ]: Tuple containing most fo the relevant information about what happened in the simulation run.
        """
        import time
        air_distance = 0.0 # Bootleg trip distance from airspeed instead of wheel speed
        transition_samples = []
        steprate = 60 # It works, don't touch it, - DON'T. TOUCH.
        stepsize = 3
        
        vehicle = self._vehicle_init()
        scenario = self._scenario_preflight(vehicle)
        self.bng.load_scenario(scenario)
        
        time.sleep(4) # Breathing room for simulator to catch up in case it hasn't
        
        self.bng.set_deterministic()
        self.bng.set_steps_per_second(steprate)
        
        limit_steps = int((steprate * truncate_seconds) / stepsize)
        
        imu_sensor, powertrain_sensor, camera = self._sensors_preflight(self.bng, vehicle)
        
        for warm_step in range(warm_up_steps):
            self.bng.step(1)
            vehicle.poll_sensors()
            try:
                raw_depth:Image.Image = camera.poll()["depth"] # Note the british version!
                damage = vehicle.sensors.__getitem__('dmg')
                elec_data = vehicle.sensors.__getitem__('elec')
                imu_data = imu_sensor.poll()
                pt_data = powertrain_sensor.poll()
            except Exception as e:
                print("Exception in warmup:\n", e)
                
        def update(step_n:int) -> tuple[dict, dict]:
            raw_depth:Image.Image = camera.poll()["depth"]
            
            try:
                vehicle.poll_sensors()
                imu_data = imu_sensor.poll()
                pt_data = powertrain_sensor.poll()
                damage = vehicle.sensors.__getitem__('dmg')
                elec_data = vehicle.sensors.__getitem__('elec')
                # NOTE: Air_trip is sampled only once every 3 physics steps
                air_trip = elec_data['airspeed']
            except:
                raise RuntimeError("Sensor fault")
            
            try:
                sight = NGSight(raw_depth)
                v_sense = NGVehicleSense(
                    airspeed=elec_data['airspeed'],
                    angular_acceleration=imu_data['angAccel'],
                    angular_velocity=imu_data['angVel'],
                    directional_acceleration=imu_data['accRaw'],
                    engine_load=elec_data['engine_load'],
                    powertrain_speed=pt_data[-1]['driveshaft']['outputAV1']
                )
                
                sight_tensor = sight.to_tensor()
                sense_tensor = v_sense.to_tensor()
                if shape_debug:
                    print("Sight shape: ", sight_tensor.shape)
                    print("Sense shape: ", sense_tensor.shape)
                
                full_data = tf.concat([sight_tensor, sense_tensor], -1)
                full_with_batch_dim = tf.expand_dims(full_data, 0) #(1, 8204), hopefully
                if shape_debug:
                    print("Shape into agent: ", full_with_batch_dim.shape)
                
                control = [
                    elec_data['steering_input']/255.0,
                    elec_data['throttle_input']-elec_data['brake_input'],
                    elec_data['parkingbrake_input']
                    ]
                
                
                    
                if step_n % sample_step_size == 0: # Check if it's a sample round.
                    # This can be taxing on memory, increase sample_step_size if necessary.
                    if len(transition_samples) > 2 and float(v_sense.airspeed) < 0.17: # Skip if too slow
                        print("Probably stuck, skipping")
                    else:
                        print("INFO: Saving sample..")
                        transition_samples.append((full_data, control))
            
            except Exception as e:
                    print("ERR! Inner update loop:", e)
            
            self.bng.step(3)
            
            try:
                return damage, elec_data, air_trip
            except:
                return {}
        
        self.bng.start_scenario()
        self.bng.pause()
        
        for current_step in range(limit_steps):
            damage_data, electrical_data, air_trip = update(current_step)
            air_distance += air_trip
            
            # "You died from bad driving" early exit
            if float(damage_data['damage']) > 4000:
                break
        
        self.bng.stop_scenario()
        #self.bng.close()
        
        # Leave moment samples as separate, pack damage, electrical and distance traveled for penalty calc use
        return transition_samples, (damage_data, electrical_data, air_distance)


    def _vehicle_init(self) -> beamngpy.Vehicle:
        from beamngpy.sensors import Damage, Electrics
        from beamngpy import Vehicle
        vehicle = Vehicle(
            'scenario_player0',
            model='bastion',
            part_config=f"{self.cfg.user_folder}/0.27/vehicles/bastion/manoever_test_car.pc"
        )
        vehicle.sensors.attach('dmg', Damage())
        vehicle.sensors.attach('elec', Electrics())
        return vehicle
    
    
    def _scenario_preflight(self, vehicle:beamngpy.Vehicle) -> None:
        scenario = self.target_scenario
        try:
            scenario.add_vehicle(vehicle, pos=self.start_pos, rot_quat=beamngpy.angle_to_quat(self.start_rot))
        except:
            print("Vehicle not added, if it is already in the scenario that's fine..")
            
        return scenario
    
    
    def _sensors_preflight(
        self,
        bng:beamngpy.BeamNGpy,
        vehicle:beamngpy.Vehicle
    ) -> tuple[
        AdvancedIMU,
        PowertrainSensor,
        Camera
    ]:  
        imu = AdvancedIMU(
            'aimu1',
            bng=bng,
            vehicle=vehicle,
            pos=(0.0, 0.0, 0.8),
            dir=(0, -1, 0),
            is_send_immediately=True,
            physics_update_time=0.05,
            is_visualised=False,
        )
        
        powertrain = PowertrainSensor('ptrn1', bng, vehicle,
            gfx_update_time=0.25,
            physics_update_time=0.05,
            is_send_immediately=False
        )
        
        camera = Camera(
            'cam1',
            bng=bng,
            vehicle=vehicle,
            is_using_shared_memory=False,
            is_render_depth=True,
            is_visualised=False,
            pos=(0, -0.25, 1.5),
            dir=(0, -1, 0),
            resolution=(256, 32), # Yes, hardcoded. At least for now.
            field_of_view_y=24.,
            near_far_planes=(0.5, 50)
        )
        return imu, powertrain, camera