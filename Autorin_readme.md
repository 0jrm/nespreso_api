I'll set up a user-level systemd service for the API that auto-restarts on crash, enable lingering if needed, start it now, and verify status.

[1 tool called]

I'll quickly inspect the API's gunicorn config and wsgi entrypoint to ensure the service starts with the right working directory and environment, then I'll create and enable the systemd user service.

[2 tools called]

I'll create a user-level systemd service for the API that runs gunicorn in your conda env, enable lingering, start it, and verify status.

[5 tools called]

Done. The API now runs as a user-level systemd service with auto-restart.

- Service file: `~/.config/systemd/user/nespreso-api.service`
- Command run: activates `nespreso` conda env, then `gunicorn -c config/gunicorn.conf.py wsgi:app`
- Restart policy: always; 5s backoff
- Lingering: enabled, so it runs without a login session

Useful commands:
- Status: `systemctl --user status nespreso-api.service`
- Logs: `journalctl --user -u nespreso-api.service -f`
- Restart: `systemctl --user restart nespreso-api.service`
- Stop: `systemctl --user stop nespreso-api.service`
- Disable: `systemctl --user disable --now nespreso-api.service`

It’s currently active and running under your conda env.

---

I'll help you change the update time from 2am to 2pm. Let me first examine the current configuration to understand where this time setting is defined.


[1 tool called]

I can see this is a README file about the API service setup, but it doesn't contain the 2am update time setting you mentioned. Let me search for files that might contain this time configuration, particularly looking for cron jobs, systemd timers, or scheduling configurations.


[4 tools called]

Let me search more specifically in the nespreso_api directory and look for Python files that might contain scheduling logic:


[2 tools called]

I can see there's a `nespreso_daily_update.sh` script that might contain the scheduling information. Let me examine that file and also look for any systemd timer files:


[2 tools called]

Let me check for systemd timers and cron jobs that might be scheduling this update:


[2 tools called]

Perfect! I found the issue. There's a cron job that runs the daily update at 2am (0 2 * * *). Let me create a todo list and then update the cron job to run at 2pm instead.


[2 tools called]

Let me try a different approach to edit the crontab. I'll create a temporary file with the updated cron entry and then install it:


[2 tools called]

Now I'll create a new crontab file with the time changed from 2am (0 2) to 2pm (0 14):


[2 tools called]

Now let me verify the change was applied correctly:


[3 tools called]

Perfect! I've successfully changed the update time from 2am to 2pm. Here's what I did:

✅ **Updated the cron job schedule**
- **Before**: `0 2 * * *` (runs at 2:00 AM daily)
- **After**: `0 14 * * *` (runs at 2:00 PM daily)

The cron job that runs your daily NeSPReSO update script (`nespreso_daily_update.sh`) will now execute at 2pm instead of 2am. The change has been verified and is now active.

The cron job will continue to:
- Run the satellite data update script with a 7-day lookback period
- Build the NeSPReSO archive
- Log all output to `/unity/g2/jmiranda/nespreso_api/logs/daily_sat_update.log`

Is there anything else you'd like me to adjust regarding the scheduling?