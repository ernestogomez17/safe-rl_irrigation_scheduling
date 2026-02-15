# Import necessary libraries
import numpy as np
import pandas as pd
import gymnasium as gym

from env.params import (
    SW, SH, SFC, S_STAR, N, ZR, KS, BETA,
    SEASON_START_DATE, LINI, LDEV, LMID, LLATE,
    KCINI, KCMID, KCEND,
)


class WaterEnvironment(gym.Env):
    metadata = {'render.modes': ['console']}

    # Soil parameters — imported from data.params
    SW = SW
    SH = SH
    SFC = SFC
    S_STAR = S_STAR
    N = N
    ZR = ZR
    KS = KS
    BETA = BETA

    # Crop parameters — imported from data.params
    SEASON_START_DATE = SEASON_START_DATE
    LINI = LINI
    LDEV = LDEV
    LMID = LMID
    LLATE = LLATE
    KCINI = KCINI
    KCMID = KCMID
    KCEND = KCEND

    def __init__(self, weather_data, n_days_ahead):
        super().__init__()
        self.spec = gym.envs.registration.EnvSpec(id="SafeWaterEnvironment-v0")
        self.n_days_ahead = n_days_ahead
        self.weather_data = weather_data
        if not pd.api.types.is_datetime64_any_dtype(self.weather_data['Date']):
            self.weather_data['Date'] = pd.to_datetime(self.weather_data['Date'])

        # Validate the range and type of self.n_days_ahead
        if not (isinstance(self.n_days_ahead, int) and 1 <= self.n_days_ahead <= 7):
            raise ValueError("The decision-making scenario (n_days_ahead) must be an integer ranging from 1 to 7 for safe irrigation practices.")
        
        # Desired starting and ending values for MAX_IRRIGATION when n_days_ahead ranges from 1 to 7
        desired_start_irrigation = 0.01  # MAX_IRRIGATION at n_days_ahead = 1
        desired_end_irrigation = 0.06     # MAX_IRRIGATION at n_days_ahead = 7

        # Corresponding range for n_days_ahead
        days_start = 1
        days_end = 7

        # Calculate the slope (a) of the line defining MAX_IRRIGATION as a function of n_days_ahead
        # Slope formula: (y2 - y1) / (x2 - x1)
        slope = (desired_end_irrigation - desired_start_irrigation) / (days_end - days_start)

        # Calculate the intercept (b) using the point-slope form of a linear equation: y - y1 = m(x - x1)
        # Rearranged to find intercept b: y = mx + b => b = y - mx
        intercept = desired_start_irrigation - (slope * days_start)

        # Adjusting the maximum irrigation based on n_days_ahead using the calculated slope and intercept
        self.MAX_IRRIGATION = slope * self.n_days_ahead + intercept


        self.action_space = gym.spaces.Box(low=np.array([0]), 
                                           high=np.array([self.MAX_IRRIGATION]), 
                                           dtype=np.float32)

        # Features: [sin_month, cos_month, sin_day, cos_day, sin_week, cos_week,
        #            soil_moisture, normalized_rho, normalized_rain]
        self.observation_space = gym.spaces.Box(
            low=np.array( [-1, -1, -1, -1, -1, -1, 0, 0, 0]),
            high=np.array([ 1,  1,  1,  1,  1,  1, 1, 1, 1]),
            dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        # Extract training mode flag
        if options is not None and 'training_mode' in options:
            self.training_mode = options['training_mode']
        else:
            self.training_mode = True  # Default to training mode

        # Use a robust seed to avoid deterministic episodes during training
        if self.training_mode:
            self.seed_value = int(np.random.randint(0, 1_000_000))
        else:
            self.seed_value = int(seed) if seed is not None else 42  # Fixed or fallback evaluation seed

        self._rng = np.random.default_rng(self.seed_value)
        np.random.seed(self.seed_value)
        self.st = self._rng.uniform(self.S_STAR, self.SFC)

        # Initialize other variables
        self.It = 0.0
        self.block_rain = 0.0   # cumulative rain over the current decision block
        self.terminated = False
        self.truncated = False
        self.elapsed_days = 0
        self.n_day_counter = 0

        # Select a random or fixed start date
        if self.training_mode:
            random_date = self._rng.choice(['2015-04-10', '2016-04-10'])
        else:
            random_date = '2017-04-10'

        start_date = pd.to_datetime(random_date)
        end_date = start_date + pd.DateOffset(years=1)

        selected_weather_data = self.weather_data[
            (self.weather_data['Date'] >= start_date) & (self.weather_data['Date'] <= end_date)
        ]

        if selected_weather_data.empty:
            raise ValueError(f"No data found for the date range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")

        self.total_days = len(selected_weather_data)

        # Initialize weather data from the selected DataFrame
        self.date_base = selected_weather_data['Date'].astype('datetime64[ns]')
        self.Rain_base = selected_weather_data['Observed Rainfall (mm)'].astype(np.float32)
        # print("Rain base: ", self.Rain_base)
        self.Tmax_base = selected_weather_data['Daily Tmax (C)'].astype(np.float32)
        self.Tmin_base = selected_weather_data['Daily Tmin (C)'].astype(np.float32)
        self.DSWR_base = selected_weather_data['Daily DSWR'].astype(np.float32)
        self.DLWR_base = selected_weather_data['Daily DLWR'].astype(np.float32)
        self.USWR_base = selected_weather_data['Daily USWR'].astype(np.float32)
        self.ULWR_base = selected_weather_data['Daily ULWR'].astype(np.float32)
        self.UGRD_base = selected_weather_data['Daily UGRD'].astype(np.float32)
        self.VGRD_base = selected_weather_data['Daily VGRD'].astype(np.float32)
        self.Pressure_base = selected_weather_data['Daily Pres (kPa)'].astype(np.float32)
        
        first_row = selected_weather_data.iloc[0]
        self.day_of_the_year = first_row['Date'].dayofyear
        self.month_of_the_year = first_row['Date'].month
        self.week_of_the_year = first_row['Date'].isocalendar()[1]
        self.current_date = first_row['Date']
        self.current_Rain = first_row['Observed Rainfall (mm)']
        self.current_Tmax = first_row['Daily Tmax (C)']
        self.current_Tmin = first_row['Daily Tmin (C)']
        self.current_DSWR = first_row['Daily DSWR']
        self.current_DLWR = first_row['Daily DLWR']
        self.current_USWR = first_row['Daily USWR']
        self.current_ULWR = first_row['Daily ULWR']
        self.current_UGRD = first_row['Daily UGRD']
        self.current_VGRD = first_row['Daily VGRD']
        self.current_Pressure = first_row['Daily Pres (kPa)']
        
        self.history_It = []
        self.history_Rain = []
        self.history_st = []
        self.history_ET_o = []
        self.history_ETmax = []
        self.history_Kc = []
        self.history_rho = []

        self.total_reward = 0.0
        self.safety_indicator = 0
        
        self.update_environment()

        self.state = self._get_obs()

        self.info = {
            "Safety indicator": np.float32(self.safety_indicator),
            "S_STAR indicator": np.float32(0),
            "SFC indicator": np.float32(0),
            "SW indicator": np.float32(0),
            "Date": self.current_date.strftime('%Y-%m-%d'),
            "ETmax": float(self.ETmax),
            "ET_o": float(self.ET_o),
            "Rain": float(self.current_Rain),
            "Tmax": float(self.current_Tmax),
            "Tmin": float(self.current_Tmin),
            "DSWR": float(self.current_DSWR),
            "DLWR": float(self.current_DLWR),
            "USWR": float(self.current_USWR),
            "ULWR": float(self.current_ULWR),
            "UGRD": float(self.current_UGRD),
            "VGRD": float(self.current_VGRD),
            "Total Days": float(self.total_days),
            "Elapsed Days": float(self.elapsed_days)
        }

        return self.state, self.info


    def advance_simulation(self):
        """Load weather data for the current day and advance the day counter.

        Returns True if data was loaded, False if the dataset is exhausted.
        History is NOT recorded here — it is recorded in step() after
        update_environment() so that all logged quantities are consistent.
        """
        if self.elapsed_days >= self.total_days:
            return False

        self.current_date = self.date_base.iloc[self.elapsed_days]
        self.current_Rain = self.Rain_base.iloc[self.elapsed_days]
        self.current_Tmax = self.Tmax_base.iloc[self.elapsed_days]
        self.current_Tmin = self.Tmin_base.iloc[self.elapsed_days]
        self.current_DSWR = self.DSWR_base.iloc[self.elapsed_days]
        self.current_DLWR = self.DLWR_base.iloc[self.elapsed_days]
        self.current_USWR = self.USWR_base.iloc[self.elapsed_days]
        self.current_ULWR = self.ULWR_base.iloc[self.elapsed_days]
        self.current_UGRD = self.UGRD_base.iloc[self.elapsed_days]
        self.current_VGRD = self.VGRD_base.iloc[self.elapsed_days]
        self.current_Pressure = self.Pressure_base.iloc[self.elapsed_days]

        self.elapsed_days += 1
        return True

    def _record_history(self):
        """Append current-day quantities to history lists.

        Called AFTER advance_simulation + update_environment so all values
        (Rain with noise, ET_o, ETmax, Kc, rho) correspond to the same day.
        Soil moisture (st) is recorded BEFORE update_soil_moisture, reflecting
        the state the agent observed when choosing an action.
        """
        self.history_Rain.append(float(self.current_Rain))
        self.history_st.append(float(self.st))
        self.history_It.append(float(self.It))
        self.history_ET_o.append(float(self.ET_o))
        self.history_ETmax.append(float(self.ETmax))
        self.history_Kc.append(float(self.Kc_value))
        self.history_rho.append(float(self.current_rho))
    
    # Function to estimate Evapotranspiration
    def calculate_ET_o(self):
        # Step 1: Mean daily temperature (Celsius)
        tmean_daily = (self.current_Tmax + self.current_Tmin) / 2
        # Step 2:  Mean daily solar radiation (Rs)
        Rs_w = (self.current_DSWR + self.current_DLWR - self.current_USWR - self.current_ULWR)
        #Rs_w = daily_dswr
        Rs = Rs_w * 0.0864  # Convert to MJ/m^2/day
        # Step 3: Wind speed (u2)
        u2 = (np.sqrt(self.current_UGRD**2 + self.current_VGRD**2) * 4.87) / np.log(67.8 * 10 - 5.42) # Adjusted wind speed at 2m
        # Assuming pressure is given in Pascals (Pa) and needs to be in kiloPascals (kPa) for the psychrometric constant calculation
        delta = 4098 * (0.6108 * np.exp((17.27 * tmean_daily) / (tmean_daily + 237.3))) / ((tmean_daily + 237.3) ** 2)
        # Step 5: Atmospheric Pressure (P) - skip be because we have pres_daily
        # Step 6: Psychometric constant
        psy = 0.000665 * self.current_Pressure
        # Step 7: Delta Term (DT)
        DT = delta / (delta + psy * (1 + 0.34 * u2))
        # Step 8: Psi Term (PT)
        PT = psy / (delta + psy * (1 + 0.34 * u2))
        # Step 9: Temperature Term (TT)
        TT = (900 * u2) / (tmean_daily + 273)
        # Step 10: Mean saturation vapor pressure derived from air temperature (es)
        et_max = 0.6108 * np.exp(17.27 * self.current_Tmax / (self.current_Tmax + 237.3))
        et_min = 0.6108 * np.exp(17.27 * self.current_Tmin / (self.current_Tmin + 237.3))
        es = (et_max + et_min) / 2
        # Step 11: Actual vapor pressure (ea) derived from relative humidity
        ea = et_min
        # Step 12: The inverse relative distance Earth-Sun (dr) and solar declination (yen)
        julian_day = self.current_date.timetuple().tm_yday
        dr = 1 + 0.033 * np.cos((2 * np.pi / 365) * julian_day)
        yen = 0.409 * np.sin((2 * np.pi / 365) * julian_day - 1.39)
        # Step 13: Conversion of latitude (AE) in degrees to radians
        Radians = np.pi * 20 / 180
        # Step 14: Sunset hour angle (…s)
        ws = np.arccos(-np.tan(Radians)*np.tan(yen))
        # Step 15: Extraterrestrial radiation (Ra)
        Ra = (24 * 60 / np.pi) * (0.0820 * dr) * ((np.sin(Radians) * np.sin(yen)) + (np.cos(Radians) * np.cos(yen) * np.sin(ws)))
        # Step 16: Clear sky solar radiation (Rso)
        Z = 602
        Rso = (0.75 + (2 * 10**-5 * Z)) * Ra
        # Step 17: Net solar or net shortwave radiation Rns
        Rns = (1 - 0.23) * Rs
        # Step 18: Net outfoing long wave solar radiation (Rnl)
        Rnl = (4.903 * 10**-9) * (((self.current_Tmax + 273.16)**4 + (self.current_Tmin + 273.16)**4) / 2) * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * (Rs/Rso) - 0.35)
        # Step 19: Net radiation (Rn)
        Rn = Rns - Rnl
        # Final steps:
        # ET Radiation
        Rng = 0.408 * Rn
        ET_rad = DT * Rng
        # ET Wind
        ET_wind = PT * TT * (es - ea)
        # Final ET_o
        ET_o = ET_rad + ET_wind
        return (ET_o)
    
    # Function to calculate the Kc value for a given day
    def Kc_function(self):
        adjusted_day = (self.day_of_the_year - self.SEASON_START_DATE) % 365
        total_growth = self.LINI + self.LDEV + self.LMID + self.LLATE
        transition_period_length = 365 - total_growth

        if adjusted_day <= self.LINI:
            return self.KCINI
        elif adjusted_day <= self.LINI + self.LDEV:
            return self.KCINI + (self.KCMID - self.KCINI) * ((adjusted_day - self.LINI) / self.LDEV)
        elif adjusted_day <= self.LINI + self.LDEV + self.LMID:
            return self.KCMID
        elif adjusted_day <= total_growth:
            return self.KCMID - (self.KCMID - self.KCEND) * ((adjusted_day - self.LINI - self.LDEV - self.LMID) / self.LLATE)
        elif adjusted_day <= total_growth + transition_period_length:
            return self.KCEND - (self.KCEND - self.KCINI) * ((adjusted_day - total_growth) / transition_period_length)
        else:
            return self.KCINI

    def calculate_ETmax(self):
        ETmax = self.Kc_value * self.ET_o
        return ETmax

    def update_environment(self):
        self.Kc_value = self.Kc_function()
        self.ET_o = self.calculate_ET_o()
        self.ETmax = self.calculate_ETmax()

        rho_value = self.rho()
        self.current_rho = rho_value.item() if isinstance(rho_value, np.ndarray) else rho_value

        if self.training_mode:
            # ±10% multiplicative noise on the ET loss rate
            self.current_rho = self.current_rho * self._rng.uniform(0.9, 1.1)

            if self.current_Rain == 0:
                # On dry days, simulate occasional missed light‐rain events.
                # A Bernoulli gate avoids injecting phantom rain every dry day,
                # which would systematically shift the seasonal water balance.
                if self._rng.random() < 0.10:  # ~10% chance of a surprise event
                    self.current_Rain = self._rng.exponential(scale=0.5)
                    self.current_Rain = min(self.current_Rain, 3.0)  # Cap at 3 mm
            else:
                # ±20% multiplicative noise — reflects typical measurement /
                # short‐range forecast uncertainty for precipitation.
                self.current_Rain = self.current_Rain * self._rng.uniform(0.80, 1.20)

    def rho(self):
        eta = self.ETmax / (self.N * self.ZR)
        Ew = 0.15 * self.ETmax
        eta_w = Ew / (self.N * self.ZR)
        m = self.KS / (self.N * self.ZR * (np.exp(self.BETA * (1 - self.SFC)) - 1))

        if self.st <= self.SH:
            return 0
        elif self.SH < self.st <= self.SW:
            return eta_w * (self.st - self.SH) / (self.SW - self.SH)
        elif self.SW < self.st <= self.S_STAR:
            return eta_w + (eta - eta_w) * (self.st - self.SW) / (self.S_STAR - self.SW)
        elif self.S_STAR < self.st <= self.SFC:
            return eta
        else:
            return eta + m * (np.exp(self.BETA * (self.st - self.SFC)) - 1)

    def update_soil_moisture(self):
        total_water = float(self.current_Rain) + (float(self.It) * 1000)
        ds_dt = (total_water / (float(self.N) * float(self.ZR))) - float(self.current_rho)
        self.st = float(self.st + ds_dt)
        self.st = max(0.0, min(1.0, self.st))
        return self.st

    def compute_reward(self):
        reward = 0.0
        it = float(self.It)

        # Ensure it is a scalar before using it in reward calculations
        if np.ndim(it) == 0:
            it = float(it)

        # Apply the irrigation penalty
        reward -= it

        return np.float32(reward)
    
    def compute_safety_indicators(self):
            # Default to safe (1) for all constraints
            is_safe_s_star, is_safe_sfc, is_safe_sw = 1.0, 1.0, 1.0

            # Check for S_STAR violation
            if self.st < self.S_STAR:
                is_safe_s_star = 0.0  # Unsafe

            # Check for SFC violation (using 'if' instead of 'elif')
            if self.st > self.SFC:
                is_safe_sfc = 0.0  # Unsafe

            # Check for SW violation
            if self.st < self.SW:
                is_safe_sw = 0.0  # Unsafe

            return np.float32(is_safe_s_star), np.float32(is_safe_sfc), np.float32(is_safe_sw)

    def _get_obs(self):
        # Sin + cos cyclical encoding for unambiguous time representation
        month_angle = 2 * np.pi * self.month_of_the_year / 12
        day_angle   = 2 * np.pi * self.day_of_the_year / 365
        week_angle  = 2 * np.pi * self.week_of_the_year / 52

        sin_month, cos_month = np.sin(month_angle), np.cos(month_angle)
        sin_day,   cos_day   = np.sin(day_angle),   np.cos(day_angle)
        sin_week,  cos_week  = np.sin(week_angle),  np.cos(week_angle)

        # Normalize rho by its theoretical maximum in the healthy operating
        # range (eta = ETmax / (N * ZR)).  Without this, rho is ~0.004–0.03
        # and the feature carries almost no dynamic range for the agent.
        eta = self.ETmax / (self.N * self.ZR) if self.ETmax > 0 else 1e-6
        normalized_rho = self.current_rho / max(eta, 1e-6)

        # Use cumulative block rain, normalized by the maximum possible
        # over n_days_ahead days so the feature stays in [0, 1].
        max_rain = float(np.max(self.Rain_base))
        normalized_rain = self.block_rain / max(max_rain * self.n_days_ahead, 1e-6)

        # Clip to valid [0, 1] range
        normalized_rho = np.clip(normalized_rho, 0, 1)
        normalized_rain = np.clip(normalized_rain, 0, 1)

        obs = np.array([
            sin_month, cos_month,
            sin_day,   cos_day,
            sin_week,  cos_week,
            self.st, normalized_rho, normalized_rain
        ], dtype=np.float32)

        return obs

    def step(self, action):
        # Ensure action is always a NumPy array, even if it's a scalar
        action = np.clip(action, self.action_space.low.item(), self.action_space.high.item())

        # Make sure it is a 1D array
        self.It = np.atleast_1d(action).astype(float)  

        # Initialize episode metrics
        self.total_reward, self.n_day_counter = 0.0, 0
        self.block_rain = 0.0

        block_is_safe_s_star = 1.0
        block_is_safe_sfc = 1.0
        block_is_safe_sw = 1.0

        # Simulate n_days_ahead days; irrigation is applied only on day 0.
        while self.n_day_counter < self.n_days_ahead:
            if self.n_day_counter > 0:
                self.It = np.array([0.0], dtype=float)

            advanced = self.advance_simulation()
            if not advanced:
                break  # dataset exhausted

            self.update_environment()
            self.block_rain += float(self.current_Rain)
            self._record_history()      # log AFTER env update so all values are same-day
            self.update_soil_moisture()

            # Update the date-based state variables
            self.day_of_the_year = self.current_date.dayofyear
            self.month_of_the_year = self.current_date.month
            self.week_of_the_year = self.current_date.isocalendar()[1]

            # Accumulate metrics
            self.total_reward += self.compute_reward()
            daily_is_safe_s_star, daily_is_safe_sfc, daily_is_safe_sw = self.compute_safety_indicators()
            block_is_safe_s_star *= daily_is_safe_s_star
            block_is_safe_sfc *= daily_is_safe_sfc
            block_is_safe_sw *= daily_is_safe_sw

            self.n_day_counter += 1

            if self.elapsed_days >= self.total_days:
                break
                

        self.safety_indicator = block_is_safe_s_star * block_is_safe_sfc * block_is_safe_sw

        # Determine if the episode is terminated or truncated
        self.terminated = False
        self.truncated = self.elapsed_days >= self.total_days

        # Prepare info for logging
        self.info = {
            "Safety indicator": np.float32(self.safety_indicator),
            "S_STAR indicator": np.float32(block_is_safe_s_star),
            "SFC indicator": np.float32(block_is_safe_sfc),
            "SW indicator": np.float32(block_is_safe_sw),
            "Date": self.current_date.strftime('%Y-%m-%d'),
            "ETmax": float(self.ETmax),
            "ET_o": float(self.ET_o),
            "Rain": float(self.current_Rain),
            "Tmax": float(self.current_Tmax),
            "Tmin": float(self.current_Tmin),
            "DSWR": float(self.current_DSWR),
            "DLWR": float(self.current_DLWR),
            "USWR": float(self.current_USWR),
            "ULWR": float(self.current_ULWR),
            "UGRD": float(self.current_UGRD),
            "VGRD": float(self.current_VGRD),
            "Total Days": float(self.total_days),
            "Elapsed Days": float(self.elapsed_days)
        }

        # Get the new state and return step outputs
        self.state = self._get_obs()
        
        return self.state, self.total_reward, self.safety_indicator, self.terminated, self.truncated, self.info


    def seed(self, seed=None):
        # Keep seeding consistent with reset
        seed_value = 42 if seed is None else int(seed)
        self._rng = np.random.default_rng(seed_value)
        np.random.seed(seed_value)