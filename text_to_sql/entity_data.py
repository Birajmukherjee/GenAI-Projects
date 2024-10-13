Entity_joins = [
    {"query": "Retrieve the list of containers whose status are in visit state active.",
     "sql": "SELECT id, visit_state, category, create_time FROM inv_unit WHERE visit_state = 'ACTIVE';"},
    {"query": "Retrieve the list of containers whose status are in departed state.",
     "sql": "SELECT id, visit_state, category, create_time FROM inv_unit WHERE visit_state = 'DEPARTED';"},
    {"query": "Provide the list of containers with goods moving from port origin to destination.",
     "sql": "SELECT id, category, visit_state, inv_unit.goods, inv_goods.origin, inv_goods.destination FROM inv_unit "
            "JOIN inv_goods ON inv_unit.goods = inv_goods.gkey WHERE visit_state = 'ACTIVE';"},
    {"query": "Retrieve the list of Horizon departed from the yard and currently in on the move towards truck.",
     "sql": "SELECT horizon, visit_state, transit_state, arrive_pos_loctype FROM inv_unit_fcy_visit WHERE "
            "arrive_pos_loctype = 'TRUCK' and transit_state = 'S70_DEPARTED';"},
    {"query": "Retrieve the list of horizon which are already departed for the transit in vessels.",
     "sql": "SELECT horizon, visit_state, transit_state, arrive_pos_loctype FROM inv_unit_fcy_visit WHERE "
            "arrive_pos_loctype = 'VESSEL' and transit_state = 'S70_DEPARTED';"},
    {"query": "Provide the list of Ships carrying vessela are on the yard and not on move.",
     "sql": "select horizon, visit_state, transit_state, arrive_pos_loctype from inv_unit_fcy_visit where "
            "arrive_pos_loctype = 'VESSEL' and transit_state = 'S40_YARD';"},
    {"query": "List out all the vessels and its names which are supposed to arrive in next 30 days in Apapa terminal.",
     "sql": "SELECT argo_carrier_visit.id as carrier_id, vsl_vessels.id as vessel_id, vsl_vessels.name as "
            "vessel_name, argo_visit_details.eta FROM"
            "vsl_vessel_visit_details JOIN argo_visit_details ON argo_visit_details.gkey = vvd_gkey JOIN "
            "argo_carrier_visit ON argo_visit_details.gkey = cvcvd_gkey JOIN vsl_vessels ON vsl_vessels.gkey = "
            "vessel_gkey WHERE eta BETWEEN NOW() AND NOW() + INTERVAL '30 days';"},
    {"query": "List out all the vessels and its names which are already departed within last 60 days from Apapa "
              "terminal.",
     "sql": "SELECT argo_carrier_visit.id as carrier_id, vsl_vessels.id as vessel_id, vsl_vessels.name as "
            "vessel_name, argo_visit_details.etd FROM"
            "vsl_vessel_visit_details JOIN argo_visit_details ON argo_visit_details.gkey = vvd_gkey JOIN "
            "argo_carrier_visit ON argo_visit_details.gkey = cvcvd_gkey JOIN vsl_vessels ON vsl_vessels.gkey = "
            "vessel_gkey WHERE etd BETWEEN NOW() - INTERVAL '60 days' AND NOW();"},
    {"query": "Fetch the count of carrier modes for each of the facility for Apapa terminal.",
     "sql": "select count(1),a.carrier_mode,b.id,b.name from argo_carrier_visit a, argo_facility b where a.fcy_gkey = "
            "b.gkey group by a.carrier_mode,b.id,b.name;"}
]