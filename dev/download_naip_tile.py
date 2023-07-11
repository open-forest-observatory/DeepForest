from pystac_client import Client
import planetary_computer as pc

# Search against the Planetary Computer STAC API
catalog = Client.open(
  "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# Define your area of interest
aoi = {
  "type": "Polygon",
  "coordinates": [
    [
      [-72.47177612354493, 44.48311174488711],
      [-72.46114564521868, 44.48311174488711],
      [-72.46114564521868, 44.48781805318961],
      [-72.47177612354493, 44.48781805318961],
      [-72.47177612354493, 44.48311174488711]
    ]
  ]
}

# Define your search with CQL2 syntax
search = catalog.search(filter_lang="cql2-json", filter={
  "op": "and",
  "args": [
    {"op": "s_intersects", "args": [{"property": "geometry"}, aoi]},
    {"op": "=", "args": [{"property": "collection"}, "naip"]}
  ]
})

# Grab the first item from the search results and sign the assets
first_item = next(search.get_items())
pc.sign_item(first_item).assets
breakpoint()

def load_planetary(url):
    if url is None:
          return None, None
    item = pystac.Item.from_file(url)
    signed_item = planetary_computer.sign(item)

    # Open one of the data assets (other asset keys to use: 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A', 'SCL', 'WVP', 'visual')
    asset_href = signed_item.assets[asset].href
    logging.info("Begining to read data")
    ds = rioxarray.open_rasterio(asset_href)
    masked_array = ds.to_masked_array()
    logging.info("Done converting data into masked array")
    masked_array = np.transpose(masked_array, (1, 2, 0))
    image = masked_array.data
    mask = np.logical_not(masked_array.mask)
    return image, mask