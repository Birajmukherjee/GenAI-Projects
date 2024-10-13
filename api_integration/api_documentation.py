def get_container_event_docs():
    return """

    ####

    API documentation: 
    Endpoint: https://api.apmterminals.com/ 
    GET /container-event-history 

    This API returns time stamped description of events registered against container ID(s). 

    ####

    Parameters table: 
    assetId | string | Either single or multiple Container IDs or Bill of Lading numbers seperated by commas | required 
    facilityCode | string | The Terminal UN Location Code. Only accepts a single Terminal Code.  USLAX for Los Angeles, USMOB for Mobile, USNWK for Port Elizabeth, USMIA for Miami, SEGOT for Gothenburg, ITVDL for Vado, NGAPP for Apapa, BHKBS for Bahrain, INNSA for Mumbai, INNSA for GTI, JOAQB for Aqaba, INPAV for Pipavav, DKAAR for Aarhus, NLRTM for Rotterdam Maasvlakte II.| required 

    ####

    Response schema (JSON object): 
    containerId | string | optional 
    shippingLine | string | optional 
    eventDetails | array[object] (Event List Result Object) 

    Each object in the "eventDetails" key has the following schema: 
    performedDateTimeLocal | string | optional 
    eventType | string | optional 
    positionNotes | string | optional 

    ####

    """

def get_import_availability_docs():
    
    return """

    ####

    API documentation: 
    Endpoint: https://api.apmterminals.com/ 

    GET /import-availability 

    This API tells whether container available for import or not at specific terminal location. 

    ####

    Query parameters table: 
    facilityCode | string | The Terminal UN Location Code to search. Only accepts a single Terminal Code.  USLAX for Los Angeles, USMOB for Mobile, USNWK for Port Elizabeth, USMIA for Miami, SEGOT for Gothenburg, ITVDL for Vado, NGAPP for Apapa, BHKBS for Bahrain, INNSA for Mumbai, INNSA for GTI, JOAQB for Aqaba, INPAV for Pipavav, DKAAR for Aarhus, NLRTM for Rotterdam Maasvlakte II.
    | required 
    assetId | string | Either single or multiple Container IDs or Bill of Lading numbers seperated by commas | required 

    ####

    Response schema (JSON object): 
    containerId | string | optional 
    billOfLading | string | optional 
    readyForDelivery | string | optional 
    containerState | string | optional 
    dischargeDateTimeLocal | string | optional
    yardLocation | string | optional 
    containerHolds | string | optional 
    storagePaidThroughDate | string | optional 
    demurrageOwed | string | optional 
    sizeTypeHeight | string | optional 
    containerIsoCode | string | optional 
    containerGrossWeightKilos | integer | optional 
    containerGrossWeightPounds | integer | optional 
    hazardous | string | optional 
    facilityOutDateTimeLocal | string | optional 
    vesselName | string | optional 
    vesselEtaDateTimeLocal | string | optional 
    shippingLine | string | optional 
    appointment | string | optional 
    appointmentDateTimeLocal | string | optional 

    ####

    """

def get_vessel_visit_docs():
    
    return """

    ####

    API documentation: 
    Endpoint: https://api.apmterminals.com/ 

    GET /vessel-visits 

    This API  returns Vessel Visits by voyage numbers 

    ####

    Query parameters table: 
    facilityCode | string | The Terminal UN Location Code to search. Only accepts a single Terminal Code. USLAX for Los Angeles, USMOB for Mobile, USNWK for Port Elizabeth, USMIA for Miami, SEGOT for Gothenburg, ITVDL for Vado, NGAPP for Apapa, BHKBS for Bahrain, INNSA for Mumbai, INNSA for GTI, JOAQB for Aqaba, INPAV for Pipavav, DKAAR for Aarhus, NLRTM for Rotterdam Maasvlakte II. | required 
    assetId | string | Either the vessel name or vessel lloyds code. | required 

    ####

    Response schema (JSON object): 
    vesselName | string | optional 
    vesselLloydsCode | string | optional 
    inboundVoyageNumber | string | optional
    outboundVoyageNumber | string | optional
    startReceiveDateTimeLocal | string | optional
    scheduledEtaDateTimeLocal | string | optional
    latestEtaDateTimeLocal | string | optional
    actualEtaDateTimeLocal | string | optional
    firstAvailableDateTimeLocal | string | optional
    scheduledEtdDateTimeLocal | string | optional
    latestEtdDateTimeLocal | string | optional
    actualEtdDateTimeLocal | string | optional
    cargoCutOffDateTimeLocal | string | optional
    reeferCutOffDateTimeLocal | string | optional
    hazardousCutoffDateTimeLocal | string | optional
    berthName | string | optional
    shippingLines | string | optional
    vesselOperator | string | optional
    vesselStatus | string | optional
    terminalDateTimeLocalStamp | string | optional 

    ####

    """

def get_all_vessel_visit_docs():

    return  """

    ####

    API documentation:
    Endpoint: https://api.apmterminals.com/

    GET /all-vessel-schedules

    This API provides the list of all vessels and their schedule for given terminal and within timerange of -6 Days and +14 days relative to current day at terminal(timezone). 

    ####

    Query parameters table:
    terminal | string | The Terminal UN Location Code to search. Only accepts a single Terminal Code.  USLAX for Los Angeles, USMOB for Mobile, USNWK for Port Elizabeth, USMIA for Miami, SEGOT for Gothenburg, ITVDL for Vado, NGAPP for Apapa, BHKBS for Bahrain, INNSA for Mumbai, INNSA for GTI, JOAQB for Aqaba, INPAV for Pipavav, DKAAR for Aarhus, NLRTM for Rotterdam Maasvlakte II.| required 

    ####

    Response schema (JSON object):
    terminalDateTimeLocalStamp | string | optional
    vesselSchedule | array[object] (Vessle Schedule Result Object)

    Each object in the "vesselSchedule" key has the following schema:
    vesselName | string | optional
    vesselLloydsCode | string | optional
    vesselOperator | string | optional
    inboundVoyageNumber | string | optional
    outboundVoyageNumber | string | optional
    scheduledArrivalDateTimeLocal | string | optional
    estimatedArrivalDateTimeLocal | string | optional
    actualArrivalDateTimeLocal | string | optional
    firstAvailableDateTimeLocal | string | optional
    scheduledDepartureDateTimeLocal | string | optional
    estimatedDepartureDateTimeLocal | string | optional
    actualDepartureDateTimeLocal | string | optional
    startReceiveDateTimeLocal | string | optional
    cargoCutoffDateTimeLocal | string | optional
    reeferCutoffDateTimeLocal | string | optional
    hazardousCutoffDateTimeLocal | string | optional
    shippingLines | array[object] (Shipping lines Result Object) 

    ####

    """

def api_docs():
    return get_import_availability_docs() + get_container_event_docs() + get_vessel_visit_docs() + get_all_vessel_visit_docs()