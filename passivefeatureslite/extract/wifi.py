def number_unique_wifi_hotspots(g):
    if g is None or len(g) == 0:
        return None
    return g["bssid"].nunique()
