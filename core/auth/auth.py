import requests
import time
import streamlit as st


class APIOAuthClient:
    def __init__(self, client_id, client_secret, token_url):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        if 'access_token' not in st.session_state:
            st.session_state.access_token = None
        if 'token_expiration' not in st.session_state:
            st.session_state.token_expiration = 0

    def get_access_token(self):
        # Return from cached token Token valid for 1hour, 60 seconds before it will accquire new token & cache it again
        if st.session_state.access_token and time.time() < st.session_state.token_expiration:
            return st.session_state.access_token
            
        token_data = self._request_new_token()
        st.session_state.access_token = token_data.get('access_token')
        expires_in = token_data.get('expires_in', 0)

        st.session_state.token_expiration = time.time() + int(expires_in) - 60
        print(f"acquired new API token, token Expires in {int(expires_in) - 60} seconds / {int(int(expires_in) - 60)/60} minutes")

        return st.session_state.access_token

    def _request_new_token(self):
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }

        response = requests.post(self.token_url, headers=headers, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get access token: {response.status_code} {response.text}")
