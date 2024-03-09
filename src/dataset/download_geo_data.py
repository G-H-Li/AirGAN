import osm2gmns as og
import pandas as pd

categories = {
    'vehicle services': ['car_wash', 'car_rental', 'vehicle_inspection', 'car_pooling', 'garage', 'garages', 'gasometer', 'parking_space','vehicle_inspection', 'bridge','toll_booth','charging_station', 'fuel', 'station', 'carport', 'waste_transfer_station', 'driver_training', 'transportation', 'hangar', 'driving_school', 'railway'],
    'transportation spots': ['boat_house', 'boathouse','train_station', 'bus_station', 'airport', 'railway_station', 'platform', 'parking', 'parking_entrance', 'marina'],
    'factories': ['factory', 'industrial', 'warehouse', 'construction', 'outbuilding', 'depot', 'barn', 'silo', 'farm_auxiliary', 'storage_tank', 'animal_breeding', 'sty', 'cowshed', 'sheepfold'],
    'decoration/furniture markets': ['marketplace', 'retail'],
    'food and beverage': ['restaurant', 'fast_food', 'cafe', 'bar', 'food_court', 'pub', 'ice_cream'],
    'shopping malls and supermarkets': ['mall', 'supermarket'],
    'sport': ['stadium', 'miniature_golf','sports_centre', 'golf_course', 'swimming_pool', 'fitness_station', 'track', 'sports_hall', 'soccer_golf', 'ice_rink', 'pitch', 'playground', 'dance', 'bicycle_parking', 'fitness_centre', 'motorcycle_parking', 'gymnasium', 'horse_riding', 'stable', 'bicycle_rental'],
    'parks': ['ruins', 'park', 'fishing', 'garden', 'statue','slipway','military', 'green_house', 'greenhouse', 'nature_reserve', 'water_park', 'public', 'grave_yard', 'farm', 'bench', 'fountain', 'outdoor_seating', 'bird_hide'],
    'culture and education': ['religious', 'theatre', 'music_school', 'mosque','pavilion', 'chapel', 'place_of_worship', 'school', 'university', 'library', 'college', 'planetarium', 'arts_centre', 'exhibition_centre', 'cinema', 'shrine', 'concert_hall', 'museum', 'temple', 'church', 'tower', 'research', 'conference_centre', 'hackerspace', 'public_bookcase', 'research_institute'],
    'entertainment': ['nightclub', 'casino', 'theme_park', 'amusement_arcade', 'social_facility', 'public_bath', 'smoking_area', 'bandstand', 'social_centre'],
    'companies': ['office', 'commercial', 'post_depot', 'postpartum_care','sanitary_dump_station','terrace','dentist', 'waste_disposal','recycling','embassy','company', 'service', 'nursing_home', 'bunker','crematorium','public_building', 'hospital', 'townhall','police', 'bank', 'fire_station', 'government', 'community_centre', 'driver_training', 'post_office', 'dam', 'clinic', 'pharmacy', 'courthouse', 'studio', 'kindergarten', 'prison', 'childcare'],
    'hotel and real estates': ['bleachers', 'detached', 'washroom','hut', 'temporary','grandstand','semidetached_house', 'common','hotel', 'apartments', 'residential', 'house', 'shelter', 'resort', 'dormitory', 'love_hotel', 'shed', 'bungalow', 'roof', 'wall', 'gate', 'gatehouse', 'beach_resort', 'toilet', 'toilets', 'ger', 'shower', 'civic']
}


def download_geo_data():
    og.downloadOSMData(912940, "../../data/beijing.osm")


def parse_link_data(path: str = "../../data/beijing.osm"):
    net = og.getNetFromFile(path,
                            link_types=['motorway', 'trunk', 'primary', 'secondary', 'tertiary'],
                            POI=True,
                            POI_sampling_ratio=1.0,
                            combine=True)
    og.consolidateComplexIntersections(net, auto_identify=True)
    og.outputNetToCSV(net, output_folder='../../data/', encoding='utf-8')


def process_link_data(file_path: str = "../../data/UrbanAir_link.csv"):
    sheet = pd.read_csv(file_path)
    sheet = sheet.drop(columns=['VDF_cap1', 'name', 'VDF_fftt1', 'allowed_uses', 'is_link', 'from_biway', 'capacity',
                                'lanes', 'free_speed', 'dir_flag', 'osm_way_id'])
    sheet.to_csv("../../data/UrbanAir_link_processed.csv")


def process_node_data(file_path: str = "../../data/UrbanAir_node.csv"):
    sheet = pd.read_csv(file_path)
    sheet = sheet.drop(columns=['name', 'osm_node_id', 'osm_highway', 'zone_id', 'ctrl_type', 'node_type',
                                'activity_type', 'is_boundary', 'intersection_id', 'poi_id', 'notes'])
    sheet.to_csv("../../data/UrbanAir_node_processed.csv")


def process_poi_data(file_path: str = "../../data/UrbanAir_poi.csv"):
    sheet = pd.read_csv(file_path)
    sheet = sheet.drop(columns=['osm_way_id', 'area_ft2', 'osm_relation_id', 'name'])
    sheet_cleaned = sheet.dropna(subset=['building', 'amenity', 'leisure', 'way'], how='all')
    sheet_cleaned['real_poi_type'] = sheet_cleaned.apply(
        lambda row: pd.Series([row['building'], row['amenity'], row['leisure'], row['way']]),
        axis=1).stack().reset_index(drop=True)
    sheet_cleaned = sheet_cleaned.drop(columns=['building', 'amenity', 'leisure', 'way'])
    sheet_cleaned = sheet_cleaned[sheet_cleaned['real_poi_type'] != 'yes']
    sheet_cleaned = sheet_cleaned[sheet_cleaned['real_poi_type'] != 'no']

    category_mapping = {item: category for category, items in categories.items() for item in items}
    sheet_cleaned['poi_type'] = sheet_cleaned['real_poi_type'].map(lambda v: category_mapping[v])
    category_id_mapping = {cate: i for i, cate in enumerate(categories.keys())}
    sheet_cleaned['poi_type_id'] = sheet_cleaned['poi_type'].map(lambda v: category_id_mapping[v])
    sheet_cleaned = sheet_cleaned.drop(columns=['real_poi_type'])
    sheet_cleaned.to_csv("../../data/UrbanAir_poi_processed.csv")


if __name__ == '__main__':
    process_poi_data()
    process_link_data()
