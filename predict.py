import pandas as pd
import feast
from joblib import load


class DriverRankingModel:
    def __init__(self):
        # Load model
        self.model = load("model.joblib")

        # Set up feature store
        self.fs = feast.FeatureStore(repo_path=".")

    def predict(self, driver_ids):
        # Read features from Feast
        driver_features = self.fs.get_online_features(
            entity_rows=[{"driver_id": driver_id} for driver_id in driver_ids],
            features=[
                "driver_hourly_stats:conv_rate",
                "driver_hourly_stats:acc_rate",
                "driver_hourly_stats:avg_daily_trips",
            ],
        )
        df = pd.DataFrame.from_dict(driver_features.to_dict())

        print("--------input------------")
        print(df[sorted(df)])
        print(df[sorted(df)].values.tolist())
        # Make prediction

        #test = [[1001, 0.2769556642, 0.120081313, 654], [1002, 0.8609133959, 0.6189895868, 207], [1003, 0.7153270841, 0.9767879248, 757]]
        test = df.values.tolist()
        #print(test)
        df["prediction"] = self.model.predict(df[sorted(df)].values.tolist())
        
        #df["prediction"] = self.model.predict(test)

        print("--------prediction------------")
        print(df)

        # Choose best driver
        best_driver_id = df["driver_id"].iloc[df["prediction"].argmax()]

        # return best driver
        return best_driver_id


if __name__ == "__main__":
    drivers = [1001, 1002, 1003]
    rows = [{"driver_id": driver_id} for driver_id in drivers]
    print(rows)
    model = DriverRankingModel()
    best_driver = model.predict(drivers)

